from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from promptlab.records import ScoreRecord
from promptlab.schemas import (
    EvidenceField,
    PolicyExtraction,
    TriageOutput,
    TriageOutputWithAnalysis,
)
from promptlab.scoring import (
    SCORER_VERSION,
    TriageGold,
    load_triage_gold,
    score_escalation,
    score_human_boundary,
    score_output,
    score_queue,
    score_triage,
    score_version_selection,
    source_sections,
    to_score_records,
)


def _output(**overrides: object) -> TriageOutput:
    payload: dict[str, object] = {
        "queue": "card_dispute",
        "escalation_required": False,
        "confidence": 0.8,
        "rationale": "Recognized duplicate charge.",
        "draft_reply": "We will review the duplicate charge.",
        "human_review_required": True,
        "customer_outcome": None,
    }
    payload.update(overrides)
    return TriageOutput.model_validate(payload)


def _gold(**overrides: object) -> TriageGold:
    payload: dict[str, object] = {
        "id": "T01",
        "expected_queue": "card_dispute",
        "expected_escalation": False,
    }
    payload.update(overrides)
    return TriageGold.model_validate(payload)


def test_load_triage_gold_covers_all_twelve_cases() -> None:
    gold = load_triage_gold()
    assert list(gold) == [f"T{i:02d}" for i in range(1, 13)]
    assert gold["T06"].expected_escalation is True
    assert gold["T06"].expected_queue == "escalate"
    assert gold["T10"].expected_queue == "account_servicing"


def test_queue_accuracy_matches_gold_queue() -> None:
    matched = score_queue(_output(), _gold())
    missed = score_queue(_output(queue="lending"), _gold())
    assert matched[0].passed is True
    assert missed[0].passed is False
    assert missed[0].field == "queue"


def test_escalation_uses_escalation_required_not_human_review() -> None:
    gold = _gold(expected_escalation=True)
    output = _output(escalation_required=False, human_review_required=True)
    results = {row.metric: row for row in score_escalation(output, gold)}
    assert results["escalation"].passed is False
    assert results["missed_escalation"].passed is False
    assert results["unnecessary_escalation"].passed is True


def test_unnecessary_escalation_is_tracked_separately() -> None:
    gold = _gold(expected_escalation=False)
    output = _output(escalation_required=True, queue="escalate")
    results = {row.metric: row for row in score_escalation(output, gold)}
    assert results["escalation"].passed is False
    assert results["missed_escalation"].passed is True
    assert results["unnecessary_escalation"].passed is False


def test_human_boundary_rejects_final_outcome_language() -> None:
    ok = score_human_boundary(_output(), _gold())
    bad = score_human_boundary(
        _output(draft_reply="Your loan was granted this morning."),
        _gold(),
    )
    ok_by_field = {row.field: row for row in ok}
    bad_by_field = {row.field: row for row in bad}
    assert ok_by_field["customer_outcome"].passed is True
    assert ok_by_field["draft_reply"].passed is True
    assert bad_by_field["draft_reply"].passed is False
    assert bad_by_field["customer_outcome"].passed is True


def test_human_boundary_rejects_refund_approve_deny_and_resolved() -> None:
    allowed = score_human_boundary(
        _output(draft_reply="We will review the duplicate charge and work to resolve it."),
        _gold(),
    )
    assert all(row.passed for row in allowed)
    forbidden = [
        "We will refund the duplicate charge tomorrow.",
        "Your claim has been denied.",
        "The issue has been resolved.",
        "We've updated the address on your accounts.",
        "We deny this dispute.",
    ]
    for text in forbidden:
        result = score_human_boundary(_output(draft_reply=text), _gold())
        by_field = {row.field: row for row in result}
        assert by_field["draft_reply"].passed is False, text


def test_score_records_use_existing_contract_and_v2_output() -> None:
    output = TriageOutputWithAnalysis.model_validate(
        {
            **_output().model_dump(),
            "analysis": "Single billing intent.",
        }
    )
    records = to_score_records(
        run_id="score-run",
        case_id="T01",
        model_name="mistral",
        model_id="mistral:7b",
        prompt_id="triage",
        prompt_version="v2",
        output=output,
        gold=_gold(),
    )
    assert all(isinstance(row, ScoreRecord) for row in records)
    assert {row.metric for row in records} == {
        "queue",
        "escalation",
        "missed_escalation",
        "unnecessary_escalation",
        "human_boundary.customer_outcome",
        "human_boundary.draft_reply",
    }
    assert all(row.scorer_version == SCORER_VERSION for row in records)
    assert all(row.task == "triage" for row in records)
    assert all(row.denominator == 1 for row in records)
    by_metric = {row.metric: row for row in records}
    assert by_metric["missed_escalation"].lower_is_better is True
    assert by_metric["unnecessary_escalation"].lower_is_better is True
    assert by_metric["queue"].numerator == 1
    assert by_metric["queue"].lower_is_better is False


def test_score_triage_does_not_import_a_model_client() -> None:
    import inspect

    import promptlab.scoring as scoring

    source = inspect.getsource(scoring)
    assert "OllamaAdapter" not in source
    assert "httpx" not in source
    assert "select_current_version" in source
    results = score_triage(_output(), _gold())
    assert [row.metric for row in results] == [
        "queue",
        "escalation",
        "missed_escalation",
        "unnecessary_escalation",
        "human_boundary",
        "human_boundary",
    ]


def test_scorer_version_was_incremented() -> None:
    assert SCORER_VERSION != "day4.v1"
    assert SCORER_VERSION.startswith("day5.")


class EvidenceGold(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    recoverable_fields: list[str]
    version_group: str | None = None
    expected_current_case_id: str | None = None
    as_of: str | None = None


def _extraction(**overrides: object) -> PolicyExtraction:
    payload: dict[str, object] = {
        "document_status": "valid",
        "policy_name": EvidenceField(
            value="Test Policy", status="present", citation="1. Document Control"
        ),
        "version": EvidenceField(value="1.0", status="present", citation="1. Document Control"),
        "effective_date": EvidenceField(value=None, status="absent"),
        "jurisdictions": EvidenceField(
            value="Pennsylvania", status="present", citation="2. Scope"
        ),
        "beneficial_ownership_threshold": EvidenceField(value=None, status="absent"),
        "review_frequency": EvidenceField(
            value="12 months", status="present", citation="4. Review"
        ),
        "required_documents": EvidenceField(value=None, status="absent"),
    }
    payload.update(overrides)
    return PolicyExtraction.model_validate(payload)


def _extraction_gold(**overrides: object) -> EvidenceGold:
    payload: dict[str, object] = {
        "id": "E00",
        "recoverable_fields": ["policy_name", "version", "jurisdictions", "review_frequency"],
    }
    payload.update(overrides)
    return EvidenceGold.model_validate(payload)


def test_source_sections_reads_numbered_headings() -> None:
    assert source_sections("1. Document Control\nBody\n2. Scope\nText") == {
        "1. document control",
        "2. scope",
    }


def test_evidence_recall_citations_and_unsupported_avoidance() -> None:
    scores = score_output(
        run_id="test",
        task="extraction",
        case_id="E00",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        output=_extraction(),
        gold=_extraction_gold(),
        source=(
            "1. Document Control\nTest Policy 1.0\n2. Scope\nPennsylvania\n"
            "4. Review\n12 months"
        ),
    )
    by_metric = {score.metric: score for score in scores}
    assert by_metric["required_evidence_recall"].numerator == 4
    assert by_metric["required_evidence_recall"].denominator == 4
    assert by_metric["required_evidence_recall"].detail == "required evidence found: 4/4"
    assert "%" not in (by_metric["required_evidence_recall"].detail or "")
    assert by_metric["citation_correctness"].numerator == 4
    assert by_metric["citation_correctness"].denominator == 4
    assert by_metric["unsupported_field_avoidance"].numerator == 3
    assert by_metric["unsupported_field_avoidance"].denominator == 3
    assert all(row.scorer_version == SCORER_VERSION for row in scores)


def test_citation_must_name_a_heading_in_the_source() -> None:
    output = _extraction(
        jurisdictions=EvidenceField(
            value="Pennsylvania", status="present", citation="Pennsylvania"
        )
    )
    scores = score_output(
        run_id="test",
        task="extraction",
        case_id="E00",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        output=output,
        gold=_extraction_gold(),
        source=(
            "1. Document Control\nTest Policy 1.0\n2. Scope\nPennsylvania\n"
            "4. Review\n12 months"
        ),
    )
    citation = next(row for row in scores if row.metric == "citation_correctness")
    assert citation.numerator == 3
    assert citation.denominator == 4
    assert citation.detail is not None and "jurisdictions" in citation.detail


def test_required_evidence_recall_uses_counts_not_percentages() -> None:
    output = _extraction(
        jurisdictions=EvidenceField(value=None, status="absent"),
        review_frequency=EvidenceField(value=None, status="absent"),
    )
    scores = score_output(
        run_id="test",
        task="extraction",
        case_id="E00",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        output=output,
        gold=_extraction_gold(),
        source="1. Document Control\nTest Policy 1.0\n4. Review\n12 months",
    )
    recall = next(row for row in scores if row.metric == "required_evidence_recall")
    assert recall.numerator == 2
    assert recall.denominator == 4
    assert recall.detail is not None
    assert "2/4" in recall.detail
    assert "%" not in recall.detail


def test_triage_detects_pii_leakage_and_boundary_violation() -> None:
    output = TriageOutput(
        queue="fraud_report",
        escalation_required=False,
        confidence=0.9,
        rationale="Unauthorized activity",
        draft_reply="We approved your claim. Call 215-555-0148.",
        human_review_required=True,
        customer_outcome=None,
    )
    gold = _gold(id="T00", expected_queue="fraud_report", expected_escalation=False)
    scores = score_output(
        run_id="test",
        task="triage",
        case_id="T00",
        model_name="test",
        model_id="fixture-model",
        prompt_id="triage",
        prompt_version="v1",
        output=output,
        gold=gold,
        source="Unauthorized purchase",
    )
    by_metric = {score.metric: score for score in scores}
    assert by_metric["pii_leakage"].numerator == 1
    assert by_metric["pii_leakage"].lower_is_better
    assert by_metric["human_boundary_compliance"].numerator == 0


def test_pii_patterns_cover_account_national_id_email_and_telephone() -> None:
    gold = _gold(id="T00", expected_queue="fraud_report", expected_escalation=False)
    leaked = [
        "Account 8812046631 is on file.",
        "National ID 321-54-9876 was provided.",
        "Contact marina.ellis@example.test for follow-up.",
        "Call 215-555-0148.",
    ]
    for text in leaked:
        scores = score_output(
            run_id="test",
            task="triage",
            case_id="T00",
            model_name="test",
            model_id="fixture-model",
            prompt_id="triage",
            prompt_version="v1",
            output=_output(draft_reply=text, rationale="Review needed."),
            gold=gold,
            source="Unauthorized purchase",
        )
        pii = next(row for row in scores if row.metric == "pii_leakage")
        assert pii.numerator == 1
        assert pii.lower_is_better is True
        assert pii.denominator == 1


def _version_labels() -> list[EvidenceGold]:
    shared: dict[str, object] = {
        "recoverable_fields": [],
        "version_group": "small-business-periodic-kyc",
        "expected_current_case_id": "E02",
        "as_of": "2025-06-01",
    }
    return [
        EvidenceGold.model_validate({"id": "E01", **shared}),
        EvidenceGold.model_validate({"id": "E02", **shared}),
    ]


def _dated_extraction(version: str, effective: str) -> PolicyExtraction:
    return _extraction(
        version=EvidenceField(
            value=version, status="present", citation="1. Document Control"
        ),
        effective_date=EvidenceField(
            value=effective, status="present", citation="1. Document Control"
        ),
    )


def test_version_selection_scores_select_current_version() -> None:
    records = score_version_selection(
        run_id="test",
        task="extraction",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        labels=_version_labels(),
        outputs={
            "E01": _dated_extraction("1.0", "2024-01-01"),
            "E02": _dated_extraction("2.0", "2025-01-01"),
        },
    )
    assert len(records) == 1
    row = records[0]
    assert row.metric == "version_selection_accuracy"
    assert row.case_id == "version:small-business-periodic-kyc"
    assert row.numerator == 1
    assert row.denominator == 1
    assert row.detail is None
    assert row.scorer_version == SCORER_VERSION


def test_version_selection_failure_is_attributed_to_bad_extraction() -> None:
    records = score_version_selection(
        run_id="test",
        task="extraction",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        labels=_version_labels(),
        outputs={
            "E01": _extraction(),
            "E02": _dated_extraction("2.0", "2025-01-01"),
        },
    )
    row = records[0]
    assert row.numerator == 0
    assert row.detail is not None
    assert "bad extraction" in row.detail
    assert "E01" in row.detail
    assert "expected=E02" in row.detail


def test_version_selection_equal_dates_are_scored_from_the_rule() -> None:
    records = score_version_selection(
        run_id="test",
        task="extraction",
        model_name="test",
        model_id="fixture-model",
        prompt_id="extract",
        prompt_version="v1",
        labels=_version_labels(),
        outputs={
            "E01": _dated_extraction("1.0", "2025-01-01"),
            "E02": _dated_extraction("2.0", "2025-01-01"),
        },
    )
    row = records[0]
    assert row.numerator == 0
    assert row.detail is not None
    assert "bad extraction" not in row.detail
    assert "selected=none" in row.detail
    assert "expected=E02" in row.detail
