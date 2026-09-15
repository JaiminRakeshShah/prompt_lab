from __future__ import annotations

import pytest
from pydantic import ValidationError

from promptlab.prompts import load
from promptlab.schemas import OUTPUT_SCHEMAS, TriageOutput, TriageOutputWithAnalysis

_BASE = {
    "queue": "card_dispute",
    "escalation_required": False,
    "confidence": 0.8,
    "rationale": "Duplicate recognized charge.",
    "draft_reply": "We will review the duplicate charge.",
    "human_review_required": True,
    "customer_outcome": None,
}


def test_v1_schema_rejects_analysis() -> None:
    with pytest.raises(ValidationError):
        TriageOutput.model_validate({**_BASE, "analysis": "short analysis"})


def test_v2_schema_requires_analysis_and_keeps_rationale() -> None:
    parsed = TriageOutputWithAnalysis.model_validate({**_BASE, "analysis": "short analysis"})
    assert parsed.rationale == _BASE["rationale"]
    assert parsed.analysis == "short analysis"
    assert set(TriageOutput.model_fields) <= set(TriageOutputWithAnalysis.model_fields)
    assert "analysis" not in TriageOutput.model_fields


def test_task_schema_map_still_points_at_base_triage() -> None:
    assert OUTPUT_SCHEMAS["triage"] is TriageOutput


def test_triage_v1_is_layered_and_does_not_request_analysis() -> None:
    template = load("triage", "v1")
    assert "queue" in template.system
    assert "escalation_required" in template.system
    assert "human_review_required" in template.system
    assert "{document_text}" not in template.system

    assert template.user_template.index("<customer_message>") < template.user_template.index(
        "{document_text}"
    )
    restated = template.user_template.split("</customer_message>", 1)[1]
    assert "TriageOutput" in restated
    assert "TriageOutputWithAnalysis" not in restated
    assert "escalation_required" in restated
    assert "analysis field" in restated.lower()


def test_triage_v2_requests_analysis_and_keeps_rationale() -> None:
    template = load("triage", "v2")
    assert "TriageOutputWithAnalysis" in template.system
    assert "analysis" in template.system
    assert "rationale" in template.system
    assert "{document_text}" not in template.system

    restated = template.user_template.split("</customer_message>", 1)[1]
    assert "TriageOutputWithAnalysis" in restated
    assert "analysis" in restated
    assert "rationale" in restated
