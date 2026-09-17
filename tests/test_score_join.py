from __future__ import annotations

from datetime import UTC, datetime

from promptlab.records import SCORE_CALL_JOIN_FIELDS, ScoreRecord, join_key
from promptlab.usage import CallRecord


def _call(**overrides: object) -> CallRecord:
    payload: dict[str, object] = {
        "record_id": "00000000-0000-4000-8000-000000000001",
        "run_id": "join-run",
        "timestamp": datetime.now(UTC),
        "provider": "ollama",
        "model_id": "qwen3:8b",
        "task": "extraction",
        "case_id": "E03",
        "prompt_id": "extract",
        "prompt_version": "v2",
        "attempt": 1,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "input_tokens": 10,
        "output_tokens": 10,
        "cached_input_tokens": None,
        "latency_ms": 1,
        "cost_usd": 0.0,
        "stop_reason": "stop",
        "error_type": None,
        "response_text": "{}",
    }
    payload.update(overrides)
    return CallRecord.model_validate(payload)


def _score(**overrides: object) -> ScoreRecord:
    payload: dict[str, object] = {
        "run_id": "join-run",
        "task": "extraction",
        "case_id": "E03",
        "model_name": "qwen",
        "model_id": "qwen3:8b",
        "prompt_id": "extract",
        "prompt_version": "v2",
        "scorer_version": "day5.v1",
        "metric": "required_evidence_recall",
        "numerator": 0,
        "denominator": 6,
    }
    payload.update(overrides)
    return ScoreRecord.model_validate(payload)


def test_score_join_fields_match_call_records() -> None:
    assert SCORE_CALL_JOIN_FIELDS == (
        "run_id",
        "case_id",
        "task",
        "model_id",
        "prompt_id",
        "prompt_version",
    )
    call = _call()
    score = _score()
    assert join_key(score) == join_key(call)


def test_score_join_does_not_fall_back_to_model_name() -> None:
    calls = [_call(), _call(model_id="mistral:7b", prompt_version="v2")]
    score = _score()
    matches = [call for call in calls if join_key(call) == join_key(score)]
    assert [call.model_id for call in matches] == ["qwen3:8b"]
