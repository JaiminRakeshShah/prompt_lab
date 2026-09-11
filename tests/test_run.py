from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.day2 import MODEL_NAMES, load_summarization_cases, main
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


def test_load_summarization_cases_returns_all_twelve() -> None:
    cases = load_summarization_cases()
    assert [case["id"] for case in cases] == [f"S{i:02d}" for i in range(1, 13)]
    assert all(case["task"] == "summarization" and case["source"] for case in cases)


def test_main_runs_all_cases_on_both_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append((self.model_id, request.case_id))
        assert request.task == "summarization"
        assert "{document_text}" not in request.user_content
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(succeeded=True, text="ok", error_type=None, records=[record])

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main()
    UUID(run_id)

    settings = Settings.from_env()
    expected = [
        (settings.models[name].model_id, f"S{i:02d}")
        for name in MODEL_NAMES
        for i in range(1, 13)
    ]
    assert calls == expected

    path = tmp_path / "runs" / f"{run_id}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 24
    parsed = [json.loads(line) for line in lines]
    assert [row["case_id"] for row in parsed] == [case_id for _, case_id in expected]
    assert [row["model_id"] for row in parsed] == [model_id for model_id, _ in expected]
    assert all(row["task"] == "summarization" for row in parsed)
    assert all(row["cost_usd"] == 0.0 for row in parsed)
