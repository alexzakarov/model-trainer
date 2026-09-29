"""Execution reward, and the preference pairs it makes possible.

This is stage 6's data half. Go is unusually well suited to preference learning
because the reward is not a model's opinion: a generated test either compiles and
passes or it does not. That makes the reward *verifiable*, which is the whole
reason preference tuning is worth doing for this domain.

Three rules keep the reward from lying:

* **Harness failures are excluded, not scored zero.** A dead container says
  nothing about the model. Scoring it zero would teach the model to avoid whatever
  it happened to be doing when the infrastructure broke.
* **The reward comes from the task's declared verification.** A trajectory is paid
  for the command the task says proves it, not for whatever command finished last.
  Otherwise a model that ends on a harmless ``read_file`` after a failing test
  looks successful.
* **A pair needs contrast.** Two samples that both pass, or both fail, are not a
  preference; inventing an order between them would train on noise. Such a task
  yields no pair, and the report says so.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .evalrun import RunConfig, Trajectory, rebuild_trajectory, trajectory_messages
from .evalstore import ResultStore
from .gorun import ExecResult, Status

#: Reward for a verification call that succeeded, and for one that ran and failed.
#: Only these two values exist: the domain's reward is binary, and a graded reward
#: would be an invention. `pass@N` supplies the variation.
SUCCESS: Final[float] = 1.0
FAILURE: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class ScoredSample:
    """One sample of a task, reduced to what preference learning needs."""

    task_id: str
    sample_index: int
    #: ``None`` when nothing gradable happened: no verification ran, or every
    #: execution was a harness error.
    reward: float | None
    verification_ran: bool
    harness_errors: int


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """A chosen and a rejected conversation for the same prompt."""

    task_id: str
    prompt: str
    chosen: Sequence[Mapping[str, Any]]
    rejected: Sequence[Mapping[str, Any]]
    chosen_reward: float
    rejected_reward: float
    margin: float

    def to_record(self) -> dict[str, Any]:
        """Serialisable form, ready for a DPO loader."""
        return {
            "prompt": self.prompt,
            "chosen": [dict(m) for m in self.chosen],
            "rejected": [dict(m) for m in self.rejected],
            "metadata": {
                "task_id": self.task_id,
                "chosen_reward": self.chosen_reward,
                "rejected_reward": self.rejected_reward,
                "margin": self.margin,
                "source": "execution",
            },
        }


@dataclass(slots=True)
class PreferenceReport:
    """What the run could and could not supply as preferences."""

    tasks: int = 0
    pairs: int = 0
    #: Why a task produced no pair, by reason.
    refusals: dict[str, int] = field(default_factory=dict)
    #: Per task, the reward of every sample that had one.
    rewards: dict[str, list[float]] = field(default_factory=dict)
    min_margin: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "tasks": self.tasks,
            "pairs": self.pairs,
            "refusals": dict(sorted(self.refusals.items())),
            "min_margin": self.min_margin,
            "rewards": dict(sorted(self.rewards.items())),
        }


def score_result(result: ExecResult) -> float | None:
    """The reward for one execution, or ``None`` when it carries no signal.

    A harness error is ``None`` rather than ``0.0``: the model was never observed
    to do anything, and a zero would be a claim about the model rather than about
    the infrastructure that broke. A ``tool_error`` and a ``model_error`` both
    score zero, because both are the model's own doing -- the command ran and said
    no, or the arguments were invalid.
    """
    if result.status is Status.HARNESS_ERROR:
        return None
    return SUCCESS if result.status is Status.OK else FAILURE


def verification_results(trajectory: Trajectory, verification: Sequence[str]) -> list[ExecResult]:
    """Every execution of a tool the task declares as its verification, in order."""
    declared = set(verification)
    return [result for result in trajectory.all_results if result.tool_name in declared]


def score_trajectory(trajectory: Trajectory, verification: Sequence[str]) -> ScoredSample:
    """Reward a trajectory by whether its declared verification last succeeded.

    The *last* call decides, because a rescue is the behaviour worth reinforcing: a
    sample that ran the tests, failed, edited, and ran them again to success should
    score the same as one that passed first time. Scoring the first call would
    reward luck and punish iteration, which is the opposite of the intent.
    """
    if not verification:
        # A task with no declared verification cannot be rewarded at all. Falling
        # back to "did anything succeed" would pay for unrelated commands.
        return ScoredSample(
            task_id=trajectory.task_id,
            sample_index=trajectory.sample_index,
            reward=None,
            verification_ran=False,
            harness_errors=sum(
                1 for r in trajectory.all_results if r.status is Status.HARNESS_ERROR
            ),
        )

    calls = verification_results(trajectory, verification)
    harness_errors = sum(1 for r in trajectory.all_results if r.status is Status.HARNESS_ERROR)
    # Drop harness failures before looking at the last call: a verification that
    # never really ran must not be read as a failed verification.
    real = [result for result in calls if result.status is not Status.HARNESS_ERROR]
    reward = score_result(real[-1]) if real else None
    return ScoredSample(
        task_id=trajectory.task_id,
        sample_index=trajectory.sample_index,
        reward=reward,
        verification_ran=bool(real),
        harness_errors=harness_errors,
    )


def _is_complete(trajectory: Trajectory) -> bool:
    """A finished trajectory ends on a turn that called no tools.

    A truncated one stops mid-loop, so its final state is not a decision the model
    made about the work; pairing it against a finished sample compares an
    interruption with a conclusion.
    """
    if trajectory.truncated or not trajectory.turns:
        return False
    return not trajectory.turns[-1].tool_calls


def pair_from_samples(
    task_id: str,
    prompt: str,
    samples: Sequence[tuple[ScoredSample, Trajectory]],
    *,
    min_margin: float = 0.0,
) -> tuple[PreferencePair | None, str]:
    """Pick the best and worst sample of a task, or say why there is no pair.

    ``min_margin`` defaults to zero, meaning any contrast counts; a caller can
    require a wider gap, and the report records the value used. With a binary
    reward a positive margin means exactly one thing: one sample's verification
    passed and another's failed.
    """
    # Narrowed here rather than in the comprehension's filter, so the reward is a
    # plain float from this point on and the arithmetic below cannot see a None.
    gradable: list[tuple[float, int, Trajectory]] = []
    for sample, trajectory in samples:
        if sample.reward is None or not _is_complete(trajectory):
            continue
        gradable.append((sample.reward, sample.sample_index, trajectory))
    if len(gradable) < 2:
        return None, "too_few_gradable_samples"

    best = max(gradable, key=lambda item: item[0])
    worst = min(gradable, key=lambda item: item[0])
    margin = best[0] - worst[0]
    if margin <= min_margin or best[1] == worst[1]:
        return None, "no_contrast"

    return (
        PreferencePair(
            task_id=task_id,
            prompt=prompt,
            chosen=trajectory_messages({"prompt": prompt}, best[2]),
            rejected=trajectory_messages({"prompt": prompt}, worst[2]),
            chosen_reward=best[0],
            rejected_reward=worst[0],
            margin=margin,
        ),
        "",
    )


def build_preferences(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    store: ResultStore,
    *,
    min_margin: float = 0.0,
) -> tuple[list[PreferencePair], PreferenceReport]:
    """Turn a run's pass@N samples into preference pairs.

    The task's ``verification`` field is what makes this possible: it names the
    tool whose outcome is the reward, so the same run that graded the model can
    also rank its own attempts.
    """
    pairs: list[PreferencePair] = []
    report = PreferenceReport(min_margin=min_margin)
    refusals: Counter[str] = Counter()

    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt", ""))
        verification = tuple(str(v) for v in (task.get("verification") or ()))
        report.tasks += 1

        if not task_id:
            refusals["missing_task_id"] += 1
            continue

        samples: list[tuple[ScoredSample, Trajectory]] = []
        for sample_index in range(config.n_samples):
            trajectory = rebuild_trajectory(config, task, sample_index, store)
            if trajectory is None:
                continue
            sample = score_trajectory(trajectory, verification)
            samples.append((sample, trajectory))

        if not any(sample.reward is not None for sample, _ in samples):
            refusals["nothing_gradable"] += 1
            continue

        report.rewards[task_id] = [
            float(sample.reward) for sample, _ in samples if sample.reward is not None
        ]
        pair, reason = pair_from_samples(task_id, prompt, samples, min_margin=min_margin)
        if pair is None:
            refusals[reason] += 1
            continue
        pairs.append(pair)

    report.pairs = len(pairs)
    report.refusals = dict(refusals)
    return pairs, report


def dpo_example(pair: PreferencePair) -> dict[str, Any]:
    """Shape a pair for a DPO loader, keeping the reward that produced it.

    The margin travels with the pair so a run can weight or filter by it later, and
    so a preference cannot be separated from the number that justified it.
    """
    return pair.to_record()
