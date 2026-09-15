from __future__ import annotations

from promptlab.records import ScoreRecord
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
from promptlab.scoring import (
    SCORER_VERSION,
    TriageGold,
    load_triage_gold,
    score_escalation,
    score_human_boundary,
    score_queue,
    score_triage,
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
    results = score_triage(_output(), _gold())
    assert [row.metric for row in results] == [
        "queue",
        "escalation",
        "missed_escalation",
        "unnecessary_escalation",
        "human_boundary",
        "human_boundary",
    ]
