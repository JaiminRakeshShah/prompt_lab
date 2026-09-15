from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.day3 import MODEL_NAME, TEMPERATURE, load_cases, main
from promptlab.usage import CallRecord


def _stub_record(request: CompletionRequest, run_id: str, model_id: str) -> CallRecord:
    return CallRecord(
        record_id="00000000-0000-4000-8000-000000000001",
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task=request.task,
        case_id=request.case_id,
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        attempt=1,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        input_tokens=1,
        output_tokens=1,
        cached_input_tokens=None,
        latency_ms=1,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=None,
        response_text="ok",
    )


def _present(value: str) -> dict[str, str]:
    return {"value": value, "status": "present", "citation": "1. Document Control"}


def _summarization_json() -> str:
    return json.dumps(
        {
            "document_status": "valid",
            "title": _present("title"),
            "version": _present("1.0"),
            "effective_date": _present("2025-01-01"),
            "purpose": _present("purpose"),
            "required_steps": _present("steps"),
            "exceptions": _present("none"),
        }
    )


def _extraction_json() -> str:
    return json.dumps(
        {
            "document_status": "valid",
            "policy_name": _present("policy"),
            "version": _present("1.0"),
            "effective_date": _present("2025-01-01"),
            "jurisdictions": _present("PA"),
            "beneficial_ownership_threshold": _present("25 percent"),
            "review_frequency": _present("24 months"),
            "required_documents": _present("formation documents"),
        }
    )


def test_load_cases_returns_all_twelve() -> None:
    summarization = load_cases("summarization.jsonl")
    extraction = load_cases("extraction.jsonl")
    assert [case["id"] for case in summarization] == [f"S{i:02d}" for i in range(1, 13)]
    assert [case["id"] for case in extraction] == [f"E{i:02d}" for i in range(1, 13)]
    assert all(case["task"] == "summarization" and case["source"] for case in summarization)
    assert all(case["task"] == "extraction" and case["source"] for case in extraction)


def test_main_runs_both_tasks_on_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str, str]] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append((self.model_id, request.task, request.case_id))
        assert request.temperature == TEMPERATURE
        assert "JSON instance" in request.system
        assert "{document_text}" not in request.user_content
        assert "{schema_description}" not in request.user_content
        if request.task == "summarization":
            assert request.prompt_id == "summarize"
            assert request.prompt_version == "v1"
            assert "SummarizationOutput" in request.user_content
            text = _summarization_json()
        else:
            assert request.prompt_id == "extract"
            assert request.prompt_version == "v2"
            assert "PolicyExtraction" in request.user_content
            assert "Northglass" in request.user_content
            text = _extraction_json()
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(succeeded=True, text=text, error_type=None, records=[record])

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main()
    UUID(run_id)

    settings = Settings.from_env()
    model_id = settings.models[MODEL_NAME].model_id
    expected = [(model_id, "summarization", f"S{i:02d}") for i in range(1, 13)] + [
        (model_id, "extraction", f"E{i:02d}") for i in range(1, 13)
    ]
    assert calls == expected

    usage_path = tmp_path / "runs" / f"{run_id}.jsonl"
    output_path = tmp_path / "runs" / f"{run_id}.outputs.jsonl"
    usage = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(usage) == 24
    assert len(outputs) == 24
    assert [row["case_id"] for row in outputs] == [case_id for _, _, case_id in expected]
    assert all(row["model_id"] == model_id for row in outputs)
    assert all(row["succeeded"] is True for row in outputs)
    assert all(row["repairs"] == 0 for row in outputs)
    assert all(row["output"] is not None for row in outputs)
    assert {row["prompt_version"] for row in outputs[:12]} == {"v1"}
    assert {row["prompt_version"] for row in outputs[12:]} == {"v2"}
