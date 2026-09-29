"""Periodic checkpoint publication to the Hugging Face Hub.

A hosted notebook run lives on a machine that is discarded when the tab closes, and
its GPU time is metered, so the only artefact that survives the run is what was
pushed. That makes "publish every N steps" an operational requirement rather than a
convenience: without it a run that dies at step 900 of 1000 has produced nothing at
all, and a 4B fine-tune is not a thing one re-runs to discover that.

Three rules, in the same spirit as the rest of this project:

**The schedule is decided before the run, not during it.** :func:`push_steps` is
pure arithmetic, so the steps a run will publish are known in advance, can be
written into the training plan, and can be asserted against :func:`should_push`,
the decision the loop actually makes. A policy that publishes "roughly every N
steps" cannot be tested; one that publishes at a fixed set of steps can.

**A token is not optional and is not discovered mid-run.** Writing to a Hub repo
requires write access, so the token is resolved when the pusher is *built* -- before
a model is loaded and before a GPU is billed. An unset variable is a clear error
naming the variable, never an unauthenticated upload that fails at step 700.

**A dry run says so, loudly.** :class:`DryRunPusher` exists so the schedule and the
upload bookkeeping can be exercised with no network at all. It is only ever selected
when the policy asks for it by name, and the run log says DRY RUN in capitals so
nobody later mistakes a rehearsal for a publication.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import DatasetError, HarnessError

#: Environment variable the token is read from by default. Named rather than
#: hard-coded into the policy so a hosted runner with a different secret name does
#: not need a code change, only a flag.
HUB_TOKEN_ENV: Final[str] = "HF_TOKEN"  # noqa: S105 - a variable name, not a secret

#: A fine-tuned checkpoint is a model repo. Stated rather than inferred at the call
#: site so the uploaded artifact type is one reviewed decision.
REPO_TYPE_MODEL: Final[str] = "model"

#: Written next to the weights so the uploaded folder describes itself: a checkpoint
#: found on the Hub says which step, which schedule and which hyperparameters
#: produced it, without needing a second system to correlate it.
PUSH_STATE_FILENAME: Final[str] = "hub_push.json"


@dataclass(frozen=True, slots=True)
class HubPushPolicy:
    """Where and how often a run publishes its checkpoint.

    ``every_steps`` counts optimiser steps, not batches: a run that accumulates
    eight micro-batches per step has taken one step, and publishing per batch would
    make the frequency depend on the accumulation setting rather than on the run.
    """

    repo_id: str
    every_steps: int
    token_env: str = HUB_TOKEN_ENV
    private: bool = False
    #: Rehearse the schedule without touching the network. Never inferred.
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Reject a policy that cannot publish, before a GPU day is spent on it."""
        if not self.repo_id.strip():
            raise HarnessError("a hub push needs a repo id; there is nowhere to push to")
        if self.every_steps < 1:
            raise HarnessError(
                f"every_steps must be >= 1 (0 would mean 'never'), got {self.every_steps}"
            )
        if not self.token_env.strip():
            raise HarnessError("a hub push needs the name of the variable holding the token")

    def describe(self) -> str:
        """One log line naming the target, the interval and whether it is a rehearsal."""
        mode = " [DRY RUN - nothing leaves this machine]" if self.dry_run else ""
        return f"HF hub {self.repo_id} every {self.every_steps} step(s){mode}"

    def to_record(self) -> dict[str, Any]:
        """Serialisable settings, for the training plan written beside the checkpoint."""
        return {
            "repo_id": self.repo_id,
            "every_steps": self.every_steps,
            "token_env": self.token_env,
            "private": self.private,
            "dry_run": self.dry_run,
        }


def push_steps(total_steps: int, every_steps: int) -> tuple[int, ...]:
    """The steps a run publishes, in order, with the final step always included.

    The final step is appended rather than left to the modulo: a run whose step
    count is not a multiple of the interval would otherwise finish without ever
    publishing the weights it actually ended on, which is the one checkpoint anyone
    is guaranteed to want.
    """
    if total_steps < 1:
        raise HarnessError(f"total_steps must be >= 1, got {total_steps}")
    if every_steps < 1:
        raise HarnessError(f"every_steps must be >= 1, got {every_steps}")
    steps = list(range(every_steps, total_steps, every_steps))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return tuple(steps)


def should_push(step: int, total_steps: int, every_steps: int) -> bool:
    """Whether the loop publishes after ``step``.

    The single decision, so :func:`push_steps` and the loop cannot disagree: the
    test asserts the two are the same predicate rather than trusting a comment.
    """
    if step < 1:
        raise HarnessError(f"step must be >= 1; step 0 is before the first update, got {step}")
    if total_steps < 1 or every_steps < 1:
        raise HarnessError(
            f"total_steps and every_steps must be >= 1, got {total_steps} and {every_steps}"
        )
    return step % every_steps == 0 or step == total_steps


def pending_final_push(last_pushed_step: int | None, step: int) -> bool:
    """Whether a finished run still owes the Hub a checkpoint.

    A run can end on a step the schedule never named -- the epoch-boundary flush
    closes a partial accumulation window, so the realised step count is not always
    the planned one. Without this the final weights would sit only on the machine
    that is about to be discarded.

    A run that took no steps at all owes nothing: its checkpoint is the untouched
    base model, and publishing it under a training repo would be a claim that
    training happened.
    """
    return step > 0 and last_pushed_step != step


def commit_message(step: int, total_steps: int) -> str:
    """The commit summary attached to a publication.

    Carries the step so a Hub history is readable as a training curve without
    opening a single file.
    """
    return f"gotooltrain step {step}/{total_steps}"


class CheckpointPusher(Protocol):
    """Publishes a checkpoint directory and names the revision it produced."""

    def push(self, directory: str | Path, step: int, total_steps: int) -> str:
        """Upload ``directory``; return an identifier for the stored revision."""


def resolve_pusher(policy: HubPushPolicy | None) -> CheckpointPusher | None:
    """The pusher a policy asks for, or ``None`` when publishing is off.

    ``None`` means "this run writes only to its own output directory", which is the
    default and is not a degraded mode: nothing is lost, the artefacts simply do not
    leave the machine.
    """
    if policy is None:
        return None
    if policy.dry_run:
        return DryRunPusher()
    return HfHubPusher(policy)


def _resolve_token(token_env: str) -> str:
    """Read the Hub token, refusing to continue without one.

    Resolved eagerly, at construction, so a missing secret is a startup error rather
    than a failure repeated once per step for the rest of the run.
    """
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise HarnessError(
            f"${token_env} is not set. Publishing to the Hugging Face Hub needs a token with "
            "write access to the target repo. Set it as a Colab secret, or pass "
            "--hub-dry-run to rehearse the schedule without uploading."
        )
    return token


def _hub_api(token: str) -> Any:  # noqa: ANN401 - huggingface_hub's own surface
    """Build a Hub client for ``token``.

    Imported lazily so this module stays importable -- and this package's own tests
    runnable -- without the training extra installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HarnessError(
            "publishing to the Hugging Face Hub needs huggingface_hub; install the 'train' extra"
        ) from exc
    return HfApi(token=token)


class HfHubPusher:
    """Uploads a checkpoint folder to a Hub model repo.

    The Hub client is injectable so the upload sequence -- create the repo if it
    does not exist, then upload the folder -- is verifiable without a network or a
    token that works.
    """

    def __init__(self, policy: HubPushPolicy, *, api: Any = None) -> None:  # noqa: ANN401
        """Bind the policy and resolve the token now, not at the first upload."""
        self.policy = policy
        self._api = api
        self._token = _resolve_token(policy.token_env)

    @property
    def identity(self) -> str:
        """The repo this pusher writes to, for the run log."""
        return self.policy.repo_id

    def _client(self) -> Any:  # noqa: ANN401 - huggingface_hub's own surface
        if self._api is None:
            self._api = _hub_api(self._token)
        return self._api

    def push(self, directory: str | Path, step: int, total_steps: int) -> str:
        """Create the repo if needed, then upload ``directory``; return the revision."""
        api = self._client()
        api.create_repo(
            self.policy.repo_id,
            repo_type=REPO_TYPE_MODEL,
            private=self.policy.private,
            exist_ok=True,
        )
        return str(
            api.upload_folder(
                folder_path=str(directory),
                repo_id=self.policy.repo_id,
                commit_message=commit_message(step, total_steps),
            )
        )


class DryRunPusher:
    """Stands in for a real upload so the schedule can be rehearsed offline.

    The revision it returns is derived from the step, so a test can assert on it
    and a log line stays meaningful -- but nothing is stored anywhere. It is
    selected only by ``HubPushPolicy(dry_run=True)``; nothing picks it by accident.
    """

    def push(self, directory: str | Path, step: int, total_steps: int) -> str:
        """Pretend to upload, naming the folder and step that would have gone."""
        del directory
        return f"dry-run-step-{step}/{total_steps}"


@dataclass(frozen=True, slots=True)
class PushRecord:
    """One publication, as it happened."""

    step: int
    total_steps: int
    #: The stored revision: a Hub commit sha, or the dry-run marker.
    revision: str
    path: str
    dry_run: bool = False

    def to_record(self) -> dict[str, Any]:
        """Serialisable form, for the run summary."""
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "revision": self.revision,
            "path": self.path,
            "dry_run": self.dry_run,
        }


def push_state_payload(
    plan_record: Mapping[str, Any], step: int, total_steps: int
) -> dict[str, Any]:
    """The self-description written into the checkpoint folder before uploading.

    Carries the whole plan, so a checkpoint found on the Hub states the
    hyperparameters, the token format and the publish schedule that produced it
    without anyone having to correlate it against a run directory.
    """
    return {
        "step": step,
        "total_steps": total_steps,
        "plan": dict(plan_record),
    }


def write_push_state(directory: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write the checkpoint's self-description, overwriting the previous step's."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / PUSH_STATE_FILENAME
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def read_push_state(directory: str | Path) -> dict[str, Any]:
    """Read the self-description a published checkpoint carries, or fail clearly."""
    path = Path(directory) / PUSH_STATE_FILENAME
    if not path.is_file():
        raise DatasetError(
            f"no {PUSH_STATE_FILENAME} in {Path(directory)}: the checkpoint does not say which "
            "run produced it"
        )
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise DatasetError(f"{path} must hold an object, got {type(decoded).__name__}")
    return decoded
