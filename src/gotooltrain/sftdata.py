"""Mine supervised training records from a completed evaluation run.

An evaluation produces trajectories. Using all of them as training data is the
mistake this module exists to prevent: a trajectory is a *record of what the model
did*, and on a task the model failed, what it did is the thing being corrected.
Training on it teaches the failure. So every trajectory is admitted or refused by
a named rule, and every refusal is counted.

The rules, and why each one is not negotiable:

* **Harness errors.** The model's behaviour was never observed. A trajectory whose
  execution failed reports the infrastructure, not the model.
* **Truncated.** The turn budget ran out while the model was still working, so the
  final turn is an interruption rather than a decision. Learning from it teaches
  the model to stop mid-task.
* **Unfinished.** Generation or execution is missing from the store, so there is no
  complete trajectory to learn from.
* **Below the score threshold.** The judge said this did not succeed.
* **No tool calls.** A conversation with no tool use teaches nothing about the
  catalogue, which is the point of the corpus.
* **Dangling calls.** A tool call with no recorded result would be rendered as a
  transcript that never occurs, and the validator refuses it anyway -- refusing it
  here names which trajectory and why.

Nothing is dropped silently: :class:`MineReport` carries a reason-keyed count and
the individual refusals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .corpus import CorpusReport, measure
from .corpus import Trajectory as MeasuredTrajectory
from .evalrun import (
    STAGE_JUDGE,
    RunConfig,
    Trajectory,
    rebuild_trajectory,
    trajectory_messages,
)
from .evalstore import ResultStore
from .gotools import GO_TOOLS, ToolDefinition
from .normalize import normalize_conversation
from .schema import Conversation
from .template import SupportsChatTemplate, render_example, require_installed_template

#: A judged trajectory is training data only at this score or above. ``1.0``
#: because the default judge awards 1 for a solved task and 0 otherwise, so a
#: lower threshold would admit exactly the failures the judge just identified.
MIN_SCORE: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class Admission:
    """Why one trajectory was or was not used, in one sentence."""

    task_id: str
    sample_index: int
    reason: str

    @property
    def admitted(self) -> bool:
        """True when there is no disqualifying reason."""
        return not self.reason


@dataclass(slots=True)
class MineReport:
    """What was mined, what was refused, and what the result can teach."""

    tasks: int = 0
    considered: int = 0
    admitted: int = 0
    #: Refusals by rule, so the shape of the loss is visible at a glance.
    refusals: dict[str, int] = field(default_factory=dict)
    #: Every refusal with its trajectory, so none is merely a number.
    refused: list[Admission] = field(default_factory=list)
    corpus: CorpusReport | None = None
    #: The threshold actually applied, recorded because it is a judgement call.
    min_score: float = MIN_SCORE

    @property
    def refused_count(self) -> int:
        """Total trajectories refused."""
        return len(self.refused)

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary for a build log."""
        return {
            "tasks": self.tasks,
            "considered": self.considered,
            "admitted": self.admitted,
            "refused": self.refused_count,
            "refusals": dict(sorted(self.refusals.items())),
            "min_score": self.min_score,
            "corpus": self.corpus.to_record() if self.corpus is not None else None,
            "refused_detail": [
                {"task_id": a.task_id, "sample_index": a.sample_index, "reason": a.reason}
                for a in self.refused
            ],
        }


def refusal_reason(
    trajectory: Trajectory,
    score: float | None,
    *,
    min_score: float,
    judged: bool,
) -> str:
    """The first rule this trajectory breaks, or an empty string when it breaks none.

    Order matters: the checks run from "we learned nothing about the model" to
    "the model did this poorly". Reporting the later reason for a harness failure
    would read as a quality problem when it is an infrastructure one.
    """
    if trajectory.has_harness_error:
        return "harness_error"
    if trajectory.truncated:
        return "truncated"
    if not trajectory.turns:
        return "empty"
    if not any(turn.tool_calls for turn in trajectory.turns):
        # A final answer with no tool use is a valid conversation but teaches
        # nothing about the catalogue, and the corpus exists to teach the catalogue.
        return "no_tool_calls"
    for turn in trajectory.turns:
        if len(turn.tool_calls) != len(turn.results):
            return "dangling_call"
    if judged:
        if score is None:
            return "unjudged"
        if score < min_score:
            return "below_threshold"
    return ""


def _measured(
    trajectory: Trajectory,
    conversation: Conversation,
    *,
    repository: str,
    package: str,
    tokenizer: SupportsChatTemplate,
) -> MeasuredTrajectory:
    """Measure one admitted trajectory the way the corpus report needs it.

    Tokens are counted by rendering the record the same way training will, so the
    supervised share in the report is the share the trainer will actually see
    rather than an estimate from character counts.
    """
    tools = tuple(
        str(call.get("name", "")) for turn in trajectory.turns for call in turn.tool_calls
    )
    example = render_example(tokenizer, conversation)
    return MeasuredTrajectory(
        trajectory_id=f"{trajectory.task_id}#{trajectory.sample_index}",
        repository=repository,
        package=package,
        tools=tools,
        supervised_tokens=example.supervised_tokens,
        total_tokens=len(example.input_ids),
        has_thinking=any(m.reasoning_content.strip() for m in conversation.messages),
        has_images=any(m.images for m in conversation.messages),
    )


def mine(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    store: ResultStore,
    tokenizer: SupportsChatTemplate,
    *,
    min_score: float = MIN_SCORE,
    judged: bool = True,
) -> tuple[list[dict[str, Any]], MineReport]:
    """Turn a finished run's trajectories into training records plus a report.

    ``judged=False`` admits on execution alone, for a run that deliberately had no
    judge. It is not a fallback for a missing verdict: passing ``judged=False`` for
    a judged run would train on trajectories the judge rejected, so the two are
    separate calls rather than one that guesses.
    """
    require_installed_template(tokenizer)
    catalogue = [_tool_spec(t) for t in GO_TOOLS]

    records: list[dict[str, Any]] = []
    report = MineReport(tasks=len(tasks), min_score=min_score)
    refusals: Counter[str] = Counter()
    measured: list[MeasuredTrajectory] = []

    for task in tasks:
        task_id = str(task.get("id") or "")
        repository = str(task.get("repository", ""))
        package = str(task.get("package", ""))
        if not task_id:
            # Refused by name rather than raising: a task with no id cannot be
            # looked up in the store, and aborting the whole mine would lose the
            # records that are perfectly good.
            for sample_index in range(config.n_samples):
                report.considered += 1
                refusals["missing_task_id"] += 1
                report.refused.append(Admission("", sample_index, "missing_task_id"))
            continue
        for sample_index in range(config.n_samples):
            report.considered += 1
            trajectory = rebuild_trajectory(config, task, sample_index, store)
            if trajectory is None:
                refusals["unfinished"] += 1
                report.refused.append(Admission(task_id, sample_index, "unfinished"))
                continue

            score: float | None = None
            if judged:
                fingerprint = config.fingerprint(task_id, sample_index, STAGE_JUDGE)
                stored = store.get(fingerprint.key)
                score = float(stored["score"]) if stored is not None else None

            reason = refusal_reason(trajectory, score, min_score=min_score, judged=judged)
            if reason:
                refusals[reason] += 1
                report.refused.append(Admission(task_id, sample_index, reason))
                continue

            messages = trajectory_messages(task, trajectory)
            conversation = normalize_conversation(messages, catalogue)
            records.append(
                {
                    "messages": messages,
                    "tools": catalogue,
                    "metadata": {
                        "task_id": task_id,
                        "sample_index": sample_index,
                        "repository": repository,
                        "package": package,
                        "score": score if score is not None else 0.0,
                        "source": "eval",
                    },
                }
            )
            measured.append(
                _measured(
                    trajectory,
                    conversation,
                    repository=repository or "unknown",
                    package=package or "unknown",
                    tokenizer=tokenizer,
                )
            )

    report.admitted = len(records)
    report.refusals = dict(refusals)
    report.corpus = measure(measured) if measured else None
    return records, report


def _tool_spec(tool: ToolDefinition) -> dict[str, Any]:
    """Render a catalogue entry in the shape the normalizer expects.

    The catalogue already stores OpenAI function-calling shape, so this is a copy
    rather than a conversion; a copy so a mined record cannot mutate the catalogue
    every later record is built from.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
