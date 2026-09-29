"""Typed errors for the tool-use data pipeline.

Every failure mode that a data engineer can plausibly hit gets its own type so
that dataset builds fail with an actionable message instead of a stack trace
from deep inside Jinja or tokenization. Validation runs *before* rendering on
purpose: a malformed record should fail during dataset construction, never
halfway through a multi-hour training run.
"""

from __future__ import annotations


class ToolTrainError(Exception):
    """Base class for every error raised by this package."""


class ValidationError(ToolTrainError):
    """A conversation does not satisfy the anthropic-tools-v1 contract."""


class UnsupportedRoleError(ValidationError):
    """A message carries a role the format does not define."""


class EmptyAssistantTurnError(ValidationError):
    """An assistant turn has no text, no thinking, and no tool call."""


class ToolCallIdError(ValidationError):
    """A tool call id is missing, duplicated, or referenced by no result."""


class DanglingToolResultError(ValidationError):
    """A tool result references a call that was never made."""


class ReservedTagError(ValidationError):
    """Untrusted content contains a reserved tag and could break out of its block.

    ``tool_result`` payloads are file contents and command output — untrusted
    input. If that text contains ``</tool_result>`` the model can be shown a
    forged block boundary. We refuse by default and offer an explicit escape
    hatch rather than silently rewriting developer data.
    """


class TemplateError(ToolTrainError):
    """The chat template failed to load or render."""


class DatasetError(ToolTrainError):
    """A dataset file is malformed or unreadable."""


class EvalStoreError(ToolTrainError):
    """The evaluation artifact store is corrupt, misused, or missing data."""


class HarnessError(ToolTrainError):
    """The execution harness lost its environment (missing binary, timeout, pool down).

    Deliberately distinct from a tool or model failure: infrastructure problems
    must never be counted as model performance.
    """


class IdempotencyViolationError(EvalStoreError):
    """One content key resolved to two different results.

    This is not a transient failure: it means the attempt fingerprint is missing
    an input that changed the outcome (a model revision, a template hash, a
    harness version). Reusing stale results under a colliding key would turn a
    regression into an apparent improvement, so the store refuses it.
    """
