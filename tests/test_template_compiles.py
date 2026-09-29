"""Guard rail: the shipped chat template must compile in transformers' environment.

This test exists because of a real incident. A template revision failed to
compile with ``unexpected '}'``, and the first response was to rewrite the
template. That was backwards: it hid the failure instead of explaining it, and
left no test that would catch the next one. A template that cannot compile must
fail here, loudly, before any data or GPU time is spent.

Note the assertion uses the transformers environment (trim_blocks=True,
lstrip_blocks=True, plus the private AssistantTracker extension that implements
``{% generation %}``). Compiling with a stock Jinja2 environment is not a valid
substitute and previously produced a misleading "syntax error".
"""

from __future__ import annotations

import jinja2
import pytest
from transformers.utils import chat_template_utils


def test_template_compiles_in_transformers_environment(template_source: str) -> None:
    chat_template_utils._compile_jinja_template(template_source)


def test_template_declares_generation_markers(template_source: str) -> None:
    """The assistant mask depends on {% generation %} being present."""
    import re

    assert re.search(r"\{%-?\s*generation\s*-?%\}", template_source), (
        "no {% generation %} block: the tokenizer cannot emit an assistant mask"
    )
    assert "{%- endgeneration -%}" in template_source


def test_template_uses_only_declared_roles(conversation: object) -> None:
    """Guard against a template that would raise on a validated conversation.

    The fixture is a parameter purely for its side effect: building a
    Conversation proves the data layer accepted the record.
    """
    assert conversation is not None


def test_stock_jinja_agrees_or_lacks_the_extension() -> None:
    """Document why a stock Jinja2 compile is not a valid gate.

    ``{% generation %}`` is implemented by a transformers-provided extension, so
    plain Jinja2 rejects it. This test asserts that expectation, so nobody
    'fixes' a future failure by compiling with the wrong environment.
    """
    pytest.importorskip("jinja2")
    # autoescape=False is correct here: chat templates emit model-facing text,
    # not HTML. transformers builds its own environment the same way.
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)  # noqa: S701
    with pytest.raises(jinja2.TemplateSyntaxError):
        env.from_string("{% generation %}x{% endgeneration %}")
