"""Template plumbing: loading, installing, exporting, and failure modes."""

from __future__ import annotations

import json
import pathlib

import pytest

from gotooltrain import (
    TemplateError,
    build_jsonl_record,
    install_template,
    load_template_source,
    render_example,
    render_text,
    require_installed_template,
    save_template_artifacts,
    template_format_version,
    template_path,
)


def test_template_is_shipped_with_the_package() -> None:
    path = template_path()
    assert path.is_file(), f"chat template missing at {path}"
    assert path.name == "anthropic-tools-v1.jinja"
    assert path.parent.name == "templates"


def test_template_source_contains_the_required_markers() -> None:
    source = load_template_source()
    assert "{% generation -%}" in source
    assert "{%- endgeneration -%}" in source
    for tag in ("<available_tools>", "<tool_use>", "<tool_result>"):
        assert tag in source


def test_template_source_is_cached() -> None:
    """Re-reading on every record would dominate dataset build time."""
    assert load_template_source() is load_template_source()


def test_install_template_is_idempotent() -> None:
    class FakeTokenizer:
        chat_template: str | None = None

    tokenizer = FakeTokenizer()
    install_template(tokenizer)
    first = tokenizer.chat_template
    install_template(tokenizer)
    assert tokenizer.chat_template == first


def test_install_template_accepts_an_explicit_source() -> None:
    class FakeTokenizer:
        chat_template: str | None = None

    tokenizer = FakeTokenizer()
    install_template(tokenizer, "custom")
    assert tokenizer.chat_template == "custom"


def test_load_reports_a_missing_template(tmp_path: pathlib.Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import gotooltrain.template as template_module

    monkeypatch.setattr(template_module, "template_path", lambda: tmp_path / "absent.jinja")
    template_module.load_template_source.cache_clear()
    try:
        with pytest.raises(TemplateError, match="chat template not found"):
            template_module.load_template_source()
    finally:
        template_module.load_template_source.cache_clear()


def test_load_rejects_a_template_without_generation_markers(
    tmp_path: pathlib.Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    import gotooltrain.template as template_module

    bad = tmp_path / "bad.jinja"
    bad.write_text("no markers here", encoding="utf-8")
    monkeypatch.setattr(template_module, "template_path", lambda: bad)
    template_module.load_template_source.cache_clear()
    try:
        with pytest.raises(TemplateError, match="no \\{% generation %\\} block"):
            template_module.load_template_source()
    finally:
        template_module.load_template_source.cache_clear()


def test_render_example_reports_a_missing_assistant_mask(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    class FakeConversation:
        def to_messages(self):  # type: ignore[no-untyped-def]
            return []

        def to_tools(self):  # type: ignore[no-untyped-def]
            return []

    class NoMaskTokenizer:
        """Carries a valid template but returns no assistant_masks."""

        chat_template = load_template_source()

        def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    with pytest.raises(TemplateError, match="assistant_masks"):
        render_example(NoMaskTokenizer(), FakeConversation())  # type: ignore[arg-type]


def test_save_template_artifacts_writes_a_verified_template(
    tmp_path: pathlib.Path, qwen_tokenizer
) -> None:  # type: ignore[no-untyped-def]
    """Transformers writes chat_template.jinja separately; losing it breaks serving."""
    written = save_template_artifacts(qwen_tokenizer, tmp_path)
    assert written.name == "chat_template.jinja"
    assert written.read_text(encoding="utf-8") == load_template_source()
    assert (tmp_path / "tokenizer_config.json").is_file()


def test_save_template_artifacts_detects_a_mismatched_template(
    tmp_path: pathlib.Path, qwen_tokenizer
) -> None:  # type: ignore[no-untyped-def]
    """A template that does not match the source must fail the export."""
    output = tmp_path / "out"
    original = qwen_tokenizer.save_pretrained

    def clobber(save_directory, **kwargs):  # type: ignore[no-untyped-def]
        original(save_directory, **kwargs)
        (pathlib.Path(save_directory) / "chat_template.jinja").write_text("nope", encoding="utf-8")

    qwen_tokenizer.save_pretrained = clobber  # type: ignore[method-assign]
    try:
        with pytest.raises(TemplateError, match="does not match"):
            save_template_artifacts(qwen_tokenizer, output)
    finally:
        qwen_tokenizer.save_pretrained = original  # type: ignore[method-assign]


def test_build_jsonl_record_includes_metadata(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    example = render_example(qwen_tokenizer, conversation)
    row = json.loads(build_jsonl_record(example, {"source": "go-ut-bench", "split": "train"}))
    assert row["format_version"] == "anthropic-tools-v1"
    assert row["metadata"] == {"source": "go-ut-bench", "split": "train"}


def test_build_jsonl_record_without_metadata(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    example = render_example(qwen_tokenizer, conversation)
    assert "metadata" not in json.loads(build_jsonl_record(example))


def test_supervised_token_count_property(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    example = render_example(qwen_tokenizer, conversation)
    assert example.supervised_tokens == sum(1 for label in example.labels if label != -100)
    assert 0 < example.supervised_tokens < len(example.input_ids)


# ------------------------------------------------------- no-fallback guarantees


def test_shipped_template_declares_the_current_format_version() -> None:
    from gotooltrain import FORMAT_VERSION

    assert template_format_version(load_template_source()) == FORMAT_VERSION


def test_unversioned_template_is_rejected() -> None:
    with pytest.raises(TemplateError, match="no format marker"):
        template_format_version("no marker here")


def test_wrong_format_version_is_rejected() -> None:
    """Extraction works on any well-formed marker; comparison is the guard's job."""
    assert template_format_version("{#-\n  format: some-other-format\n#}") == "some-other-format"


def test_tokenizer_without_template_fails_loudly() -> None:
    """No silent disk fallback: a template-less tokenizer is a packaging bug.

    Falling back would let training render with the shipped template while
    serving uses none, and the only symptom would be a model that appears to
    have lost tool use.
    """

    class BareTokenizer:
        chat_template: str | None = None

    with pytest.raises(TemplateError, match="no chat_template"):
        require_installed_template(BareTokenizer())


def test_tokenizer_with_foreign_template_fails_loudly() -> None:
    class ForeignTokenizer:
        chat_template = "{#-\n  format: some-other-format\n#}"

    with pytest.raises(TemplateError, match="disagree about the token format"):
        require_installed_template(ForeignTokenizer())


def test_render_refuses_a_template_less_tokenizer(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    class Bare:
        chat_template = None

        def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("must not reach the tokenizer")

    with pytest.raises(TemplateError, match="no chat_template"):
        render_text(Bare(), conversation)  # type: ignore[arg-type]


def test_installed_template_passes_the_guard(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    from gotooltrain import FORMAT_VERSION

    assert template_format_version(require_installed_template(qwen_tokenizer)) == FORMAT_VERSION


def test_load_rejects_a_foreign_format_version(tmp_path: pathlib.Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import gotooltrain.template as template_module

    foreign = tmp_path / "foreign.jinja"
    foreign.write_text(
        "{#-\n  format: other-format-v9\n#}\n{% generation %}x{% endgeneration %}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(template_module, "template_path", lambda: foreign)
    template_module.load_template_source.cache_clear()
    try:
        with pytest.raises(TemplateError, match="declares format"):
            template_module.load_template_source()
    finally:
        template_module.load_template_source.cache_clear()


def test_save_reports_a_tokenizer_that_writes_no_template(tmp_path: pathlib.Path) -> None:
    """No silent fallback: a missing chat_template.jinja must abort the export."""

    class SilentTokenizer:
        chat_template: str | None = None

        def save_pretrained(self, save_directory, **kwargs):  # type: ignore[no-untyped-def]
            (pathlib.Path(save_directory) / "tokenizer_config.json").write_text(
                "{}", encoding="utf-8"
            )

    with pytest.raises(TemplateError, match="was not written"):
        save_template_artifacts(SilentTokenizer(), tmp_path)  # type: ignore[arg-type]


def test_render_rejects_a_non_string_result() -> None:
    """A tokenizer that returns token ids when asked for text is a real error."""

    class Confused:
        chat_template = None

        def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return [1, 2, 3]

    with pytest.raises(TemplateError, match="no chat_template"):
        render_text(Confused(), _FakeConversation())  # type: ignore[arg-type]


class _FakeConversation:
    def to_messages(self):  # type: ignore[no-untyped-def]
        return []

    def to_tools(self):  # type: ignore[no-untyped-def]
        return []


def test_render_rejects_a_non_string_result_with_valid_template() -> None:
    class Confused:
        chat_template = "{#-\n  format: anthropic-tools-v1\n#}"

        def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return [1, 2, 3]

    with pytest.raises(TemplateError, match="expected a rendered string"):
        render_text(Confused(), _FakeConversation())  # type: ignore[arg-type]
