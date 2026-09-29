"""Workspace mutations for the tools that are not commands.

``write_file`` and ``edit_file`` have no shell equivalent: they edit the workspace
directly. Modelling them as an empty command made every call succeed while doing
nothing, which is the worst possible failure here -- the model would be trained on
"my edit succeeded" while the repository never changed, and the next turn would be
reasoning about a file that does not exist.

The security boundary is the point of this module. Paths arrive from a model, so
every path is resolved and then *re-checked* to be inside the workspace after
symlink resolution. Rejecting the string ``../etc/passwd`` is not enough: a
symlink inside the workspace can point anywhere, so containment is verified on the
resolved path rather than the requested one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import ValidationError

#: Refuse to read a whole file into memory beyond this; a model that asks for a
#: 4 GB file has made a mistake, and honouring it would stall a worker.
MAX_FILE_BYTES: Final[int] = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EditOutcome:
    """Result of a workspace mutation, shaped like a tool result."""

    #: Short human-readable confirmation, fed back to the model.
    message: str
    #: False when the model asked for something that did not apply.
    applied: bool


def resolve_in_workspace(workspace: Path, requested: str) -> Path:
    """Resolve a model-supplied relative path, refusing to leave the workspace.

    Containment is checked on the resolved path, so neither ``..`` segments, an
    absolute path, nor a symlink pointing outside the workspace can be used to
    escape. A violation is a *model* error: the arguments were wrong.
    """
    if not requested or not requested.strip():
        raise ValidationError("path must not be empty")
    if Path(requested).is_absolute() or (len(requested) > 1 and requested[1] == ":"):
        raise ValidationError(f"path must be repo-relative, got absolute path {requested!r}")

    root = workspace.resolve()
    target = (root / requested).resolve()
    if target != root and root not in target.parents:
        raise ValidationError(f"path escapes the workspace: {requested!r}")
    return target


def write_file(workspace: Path, arguments: dict[str, Any]) -> EditOutcome:
    r"""Create or overwrite a file, creating parent directories as needed.

    Line endings are normalised to ``\\n``. A model that emits CRLF would
    otherwise produce a file that ``gofmt`` rewrites and ``go build`` reads
    differently from what the model was shown, and the next turn's diagnostics
    would not line up with the file it thinks it wrote.
    """
    target = resolve_in_workspace(workspace, str(arguments.get("path", "")))
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValidationError("write_file.content must be a string")
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")

    existed = target.is_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalised, encoding="utf-8", newline="\n")
    lines = normalised.count("\n") + (0 if normalised.endswith("\n") or not normalised else 1)
    verb = "updated" if existed else "created"
    return EditOutcome(f"{verb} {arguments['path']} ({lines} lines, {len(normalised)} bytes)", True)


def edit_file(workspace: Path, arguments: dict[str, Any]) -> EditOutcome:
    """Replace a unique exact string.

    The uniqueness requirement is enforced, not advisory. Silently editing the
    first of several matches is how an agent "fixes" the wrong function and never
    learns why; the error is returned to the model so it can add context.
    """
    target = resolve_in_workspace(workspace, str(arguments.get("path", "")))
    old = arguments.get("old_string")
    new = arguments.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValidationError("edit_file requires string old_string and new_string")
    if not target.is_file():
        raise ValidationError(f"cannot edit missing file {arguments['path']!r}")
    if len(target.read_bytes()) > MAX_FILE_BYTES:
        raise ValidationError(f"file is too large to edit: {arguments['path']!r}")
    if old == new:
        raise ValidationError("old_string and new_string are identical; nothing to do")

    original = target.read_text(encoding="utf-8")
    occurrences = original.count(old)
    if occurrences == 0:
        raise ValidationError(
            f"old_string not found in {arguments['path']!r}; read the file again to get the "
            "current text (it may differ in whitespace)"
        )
    if occurrences > 1:
        raise ValidationError(
            f"old_string appears {occurrences} times in {arguments['path']!r}; include more "
            "surrounding context to make it unique"
        )

    updated = original.replace(old, new, 1)
    target.write_text(updated, encoding="utf-8", newline="\n")
    changed = updated.count("\n") - original.count("\n")
    return EditOutcome(f"edited {arguments['path']} (1 replacement, {changed:+d} lines)", True)
