"""Publication to the Hub: the schedule, the guardrails, and the upload.

Two things here are worth a test beyond "it did not raise":

* **The schedule and the loop must agree.** :func:`push_steps` computes the steps in
  advance and :func:`should_push` is what the training loop actually calls. They are
  two spellings of one decision, so the test asserts they are the same predicate
  across a grid of intervals rather than trusting the docstrings to be right.
* **A missing token is a startup error, not an upload that fails later.** The
  expensive failure mode is a run that trains for hours and then discovers the
  secret was never set, so the refusal is asserted to happen at construction.

Nothing here touches the network: the Hub client is injectable, and the real client
factory is exercised only as far as building an object goes.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain import hub
from gotooltrain.errors import DatasetError, HarnessError
from gotooltrain.hub import (
    HUB_TOKEN_ENV,
    PUSH_STATE_FILENAME,
    REPO_TYPE_MODEL,
    DryRunPusher,
    HfHubPusher,
    HubPushPolicy,
    PushRecord,
    commit_message,
    pending_final_push,
    push_state_payload,
    push_steps,
    read_push_state,
    resolve_pusher,
    should_push,
    write_push_state,
)

MOCK_TOKEN = "hf_mock_token_for_local_tests_not_a_real_credential"

#: A repository id that does not exist, on purpose. The real publication target lives
#: in one constant, ``build_notebook.DEFAULT_HF_REPO_ID``; a reader who copies a repo
#: out of a test should be unable to publish anything by accident.
FIXTURE_REPO = "example/not-a-real-repo"


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    """A token in the environment, so a pusher can be built at all."""
    monkeypatch.setenv(HUB_TOKEN_ENV, MOCK_TOKEN)
    return MOCK_TOKEN


def policy(**overrides: Any) -> HubPushPolicy:
    base: dict[str, Any] = {"repo_id": FIXTURE_REPO, "every_steps": 2}
    base.update(overrides)
    return HubPushPolicy(**base)


class FakeApi:
    """A Hub client that records calls instead of making them."""

    def __init__(self) -> None:
        """Start with no calls made."""
        self.created: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

    def create_repo(self, repo_id: str, **kwargs: Any) -> str:
        """Record a repo creation and answer with a plausible url."""
        self.created.append({"repo_id": repo_id, **kwargs})
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, **kwargs: Any) -> str:
        """Record an upload and answer with a plausible commit sha."""
        self.uploads.append(dict(kwargs))
        return "0" * 40


# ----------------------------------------------------------------- the policy


def test_a_policy_names_its_target_and_interval() -> None:
    described = policy().describe()
    assert FIXTURE_REPO in described
    assert "every 2 step" in described


def test_a_dry_run_says_so_in_the_log_line() -> None:
    """A rehearsal that reads like a real upload is worse than no rehearsal."""
    assert "DRY RUN" in policy(dry_run=True).describe()
    assert "DRY RUN" not in policy().describe()


def test_a_policy_is_recorded_whole() -> None:
    record = policy(token_env="COLAB_HF", private=True).to_record()
    assert record == {
        "repo_id": FIXTURE_REPO,
        "every_steps": 2,
        "token_env": "COLAB_HF",
        "private": True,
        "dry_run": False,
    }


def test_a_policy_with_nowhere_to_push_to_is_refused() -> None:
    with pytest.raises(HarnessError, match="nowhere to push to"):
        policy(repo_id="  ")


def test_an_interval_of_zero_is_refused_rather_than_meaning_never() -> None:
    """0 would be indistinguishable from 'publishing was never configured'."""
    with pytest.raises(HarnessError, match="every_steps must be >= 1"):
        policy(every_steps=0)


def test_a_policy_with_no_token_variable_is_refused() -> None:
    with pytest.raises(HarnessError, match="variable holding the token"):
        policy(token_env="")


# ------------------------------------------------------------- the schedule


def test_the_schedule_lands_on_the_interval() -> None:
    assert push_steps(total_steps=6, every_steps=2) == (2, 4, 6)


def test_the_final_step_is_published_even_off_the_interval() -> None:
    """The weights the run ended on are the ones anyone is guaranteed to want."""
    assert push_steps(total_steps=7, every_steps=2) == (2, 4, 6, 7)


def test_an_interval_equal_to_the_run_publishes_once_not_twice() -> None:
    assert push_steps(total_steps=10, every_steps=10) == (10,)


def test_a_run_shorter_than_the_interval_still_publishes() -> None:
    assert push_steps(total_steps=3, every_steps=50) == (3,)


def test_the_loop_and_the_schedule_are_the_same_predicate() -> None:
    """The decision the loop makes must be the one the plan advertises.

    Checked over a grid rather than one case: a schedule that drifts from the loop
    is the kind of divergence that passes a single example and then skips every
    intermediate publication of a long run.
    """
    for total_steps in range(1, 13):
        for every_steps in range(1, 6):
            planned = set(push_steps(total_steps, every_steps))
            observed = {
                s for s in range(1, total_steps + 1) if should_push(s, total_steps, every_steps)
            }
            assert observed == planned, (total_steps, every_steps, observed, planned)


def test_a_step_before_the_first_update_is_refused() -> None:
    """Step 0 is the untouched base model; publishing it claims training happened."""
    with pytest.raises(HarnessError, match="before the first update"):
        should_push(0, 10, 2)


def test_a_zero_step_budget_is_refused() -> None:
    with pytest.raises(HarnessError, match="total_steps must be >= 1"):
        push_steps(0, 2)
    with pytest.raises(HarnessError, match="must be >= 1"):
        should_push(1, 0, 2)
    with pytest.raises(HarnessError, match="must be >= 1"):
        push_steps(10, 0)
    with pytest.raises(HarnessError, match="must be >= 1"):
        should_push(1, 10, 0)


def test_a_finished_run_publishes_what_the_schedule_missed() -> None:
    """A partial accumulation window can land the run on an unnamed step."""
    assert pending_final_push(last_pushed_step=2, step=3) is True
    assert pending_final_push(last_pushed_step=3, step=3) is False
    assert pending_final_push(last_pushed_step=None, step=3) is True


def test_a_run_that_took_no_steps_owes_nothing() -> None:
    assert pending_final_push(last_pushed_step=None, step=0) is False


# ---------------------------------------------------------------- the pusher


def test_no_policy_means_no_pusher() -> None:
    """Publishing off is a mode, not a failure: nothing is lost, nothing is sent."""
    assert resolve_pusher(None) is None


def test_a_dry_run_policy_selects_the_dry_run_pusher() -> None:
    assert isinstance(resolve_pusher(policy(dry_run=True)), DryRunPusher)


def test_a_real_policy_builds_a_hub_pusher(token: str) -> None:
    assert isinstance(resolve_pusher(policy()), HfHubPusher)


def test_a_missing_token_stops_the_run_before_a_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4B fine-tune is not a thing to re-run to discover the secret was unset."""
    monkeypatch.delenv(HUB_TOKEN_ENV, raising=False)
    with pytest.raises(HarnessError, match=f"\\${HUB_TOKEN_ENV} is not set"):
        resolve_pusher(policy())


def test_a_blank_token_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HUB_TOKEN_ENV, "   ")
    with pytest.raises(HarnessError, match="is not set"):
        resolve_pusher(policy())


def test_the_token_can_come_from_a_named_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted runner with a different secret name needs a flag, not a patch."""
    monkeypatch.delenv(HUB_TOKEN_ENV, raising=False)
    monkeypatch.setenv("COLAB_HF", MOCK_TOKEN)
    assert resolve_pusher(policy(token_env="COLAB_HF")) is not None


def test_the_upload_creates_the_repo_then_sends_the_folder(
    tmp_path: pathlib.Path, token: None
) -> None:
    api = FakeApi()
    pusher = HfHubPusher(policy(private=True), api=api)

    revision = pusher.push(tmp_path, step=4, total_steps=10)

    assert revision == "0" * 40
    assert api.created == [
        {
            "repo_id": FIXTURE_REPO,
            "repo_type": REPO_TYPE_MODEL,
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploads == [
        {
            "folder_path": str(tmp_path),
            "repo_id": FIXTURE_REPO,
            "commit_message": "gotooltrain step 4/10",
        }
    ]


def test_the_pusher_names_its_repo(token: None) -> None:
    assert HfHubPusher(policy()).identity == FIXTURE_REPO


def test_a_pusher_without_an_injected_client_builds_one(
    tmp_path: pathlib.Path, token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path is the real one, so it is exercised once with a stub factory."""
    api = FakeApi()
    monkeypatch.setattr(hub, "_hub_api", lambda _token: api)

    HfHubPusher(policy()).push(tmp_path, step=1, total_steps=1)

    assert len(api.uploads) == 1


def test_the_real_client_factory_builds_a_hub_api() -> None:
    """Construction is offline; this pins that the import and the call still work."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    assert isinstance(hub._hub_api(MOCK_TOKEN), huggingface_hub.HfApi)


def test_a_dry_run_upload_touches_nothing_on_disk(tmp_path: pathlib.Path) -> None:
    """The whole point: a rehearsal must leave no trace where a real one would."""
    before = sorted(p.name for p in tmp_path.iterdir())
    revision = DryRunPusher().push(tmp_path, step=3, total_steps=9)
    assert revision == "dry-run-step-3/9"
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------- the record


def test_a_push_record_serialises() -> None:
    record = PushRecord(step=2, total_steps=9, revision="abc", path="/out", dry_run=True)
    assert record.to_record() == {
        "step": 2,
        "total_steps": 9,
        "revision": "abc",
        "path": "/out",
        "dry_run": True,
    }


def test_the_commit_message_carries_the_step() -> None:
    assert commit_message(5, 10) == "gotooltrain step 5/10"


# ------------------------------------------------------- the pushed folder


def test_the_folder_describes_itself(tmp_path: pathlib.Path) -> None:
    """A checkpoint on the Hub must say which run produced it, unaided."""
    plan_record = {"model_id": "Qwen/Qwen3.5-4B", "token_format": "anthropic-tools-v1"}
    path = write_push_state(tmp_path, push_state_payload(plan_record, step=4, total_steps=10))

    assert path.name == PUSH_STATE_FILENAME
    assert read_push_state(tmp_path) == {
        "step": 4,
        "total_steps": 10,
        "plan": plan_record,
    }


def test_describing_the_folder_creates_it(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run" / "checkpoint"
    write_push_state(target, push_state_payload({}, step=1, total_steps=1))
    assert (target / PUSH_STATE_FILENAME).is_file()


def test_the_previous_steps_description_is_replaced(tmp_path: pathlib.Path) -> None:
    write_push_state(tmp_path, push_state_payload({}, step=1, total_steps=10))
    write_push_state(tmp_path, push_state_payload({}, step=5, total_steps=10))
    assert read_push_state(tmp_path)["step"] == 5


def test_a_folder_with_no_description_is_reported_as_such(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="does not say which run produced it"):
        read_push_state(tmp_path)


def test_a_corrupt_description_is_reported_as_corrupt(tmp_path: pathlib.Path) -> None:
    (tmp_path / PUSH_STATE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="is not valid JSON"):
        read_push_state(tmp_path)


def test_a_description_that_is_not_an_object_is_refused(tmp_path: pathlib.Path) -> None:
    (tmp_path / PUSH_STATE_FILENAME).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DatasetError, match="must hold an object"):
        read_push_state(tmp_path)


def test_the_description_is_written_with_a_trailing_newline(tmp_path: pathlib.Path) -> None:
    """Byte-stable so a diff of two steps is a diff of content, not of newline."""
    write_push_state(tmp_path, push_state_payload({"a": 1}, step=1, total_steps=2))
    body = (tmp_path / PUSH_STATE_FILENAME).read_text(encoding="utf-8")
    assert body.endswith("\n")
    assert json.loads(body)["step"] == 1


def test_the_description_does_not_alias_the_caller_mapping(tmp_path: pathlib.Path) -> None:
    """A later mutation of the plan must not rewrite what was published."""
    plan_record = {"model_id": "a"}
    payload = push_state_payload(plan_record, step=1, total_steps=1)
    plan_record["model_id"] = "b"
    write_push_state(tmp_path, payload)
    assert read_push_state(tmp_path)["plan"]["model_id"] == "a"


def test_the_token_never_reaches_the_published_state(tmp_path: pathlib.Path) -> None:
    """A secret in an artefact that lives on the Hub is a leaked secret."""
    plan_record = {"model_id": "a", "token_env": HUB_TOKEN_ENV}
    write_push_state(tmp_path, push_state_payload(plan_record, step=1, total_steps=1))
    assert MOCK_TOKEN not in (tmp_path / PUSH_STATE_FILENAME).read_text(encoding="utf-8")


def test_the_token_is_read_from_the_environment_not_hard_coded() -> None:
    """The shipped constant is a variable *name*; a literal credential would be a leak."""
    assert HUB_TOKEN_ENV == "HF_TOKEN"
    source = pathlib.Path(hub.__file__).read_text(encoding="utf-8")
    assert MOCK_TOKEN not in source, "a credential was written into the module"
