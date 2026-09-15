from __future__ import annotations

import pytest
from pydantic import BaseModel

from promptlab.config import PROJECT_ROOT
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    render_prompt,
    schema_description,
)

PROMPTS = (
    ("summarize.v1.md", SummarizationOutput),
    ("extract.v1.md", PolicyExtraction),
    ("extract.v2.md", PolicyExtraction),
)


def test_schema_description_is_json_schema_from_the_model() -> None:
    text = schema_description(SummarizationOutput)
    assert '"title": "SummarizationOutput"' in text
    assert "citation" in text
    assert "{" in text and "}" in text


def test_format_crashes_once_schema_text_is_in_the_prompt() -> None:
    injected = "{schema_description} {document_text}".replace(
        "{schema_description}", schema_description(PolicyExtraction)
    )
    with pytest.raises(KeyError):
        injected.format(document_text="BODY")

    rendered = render_prompt("{schema_description}\n{document_text}", PolicyExtraction, "BODY")
    assert rendered.endswith("BODY")
    assert "{schema_description}" not in rendered


@pytest.mark.parametrize(("filename", "model"), PROMPTS)
def test_day3_prompts_consume_generated_schema(filename: str, model: type[BaseModel]) -> None:
    template = (PROJECT_ROOT / "src" / "prompts" / filename).read_text(encoding="utf-8")
    rendered = render_prompt(template, model, "CASE_BODY")
    assert "{schema_description}" not in rendered
    assert "{document_text}" not in rendered
    assert "CASE_BODY" in rendered
    assert "citation" in rendered
    assert model.__name__ in rendered
