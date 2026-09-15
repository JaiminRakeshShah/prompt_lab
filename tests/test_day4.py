from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.day4 import (
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    build_request,
    load_triage_cases,
    main,
    run_case,
)
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
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


def _triage_json(*, with_analysis: bool) -> str:
    payload: dict[str, object] = {
        "queue": "card_dispute",
        "escalation_required": False,
        "confidence": 0.8,
        "rationale": "Recognized duplicate charge.",
        "draft_reply": "We will review the duplicate charge.",
        "human_review_required": True,
        "customer_outcome": None,
    }
    if with_analysis:
        payload["analysis"] = "Single billing intent; not fraud."
    return json.dumps(payload)


def test_load_triage_cases_returns_all_twelve() -> None:
    cases = load_triage_cases()
    assert [case["id"] for case in cases] == [f"T{i:02d}" for i in range(1, 13)]
    assert all(case["task"] == "triage" and case["source"] for case in cases)


def test_main_compares_prompt_versions_on_one_mistral_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[CompletionRequest] = []
    model_ids: list[str] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append(request)
        model_ids.append(self.model_id)
        assert request.task == "triage"
        assert request.prompt_id == "triage"
        assert request.temperature == TEMPERATURE == 0.0
        assert request.max_output_tokens == MAX_OUTPUT_TOKENS
        assert request.system
        assert "JSON instance" in request.system
        assert "{document_text}" not in request.user_content
        assert "<customer_message>" in request.user_content
        assert request.case_id not in request.system
        if request.prompt_version == "v1":
            assert "JSON instance of TriageOutput." in request.system
            assert "TriageOutputWithAnalysis" not in request.system
            assert "analysis field" in request.user_content.lower()
            text = _triage_json(with_analysis=False)
        else:
            assert request.prompt_version == "v2"
            assert "JSON instance of TriageOutputWithAnalysis." in request.system
            assert "analysis" in request.user_content
            text = _triage_json(with_analysis=True)
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(succeeded=True, text=text, error_type=None, records=[record])

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main()
    UUID(run_id)

    settings = Settings.from_env()
    model_id = settings.models[MODEL_NAME].model_id
    assert model_id == settings.models["mistral"].model_id
    assert model_id != settings.models["qwen"].model_id
    assert set(model_ids) == {model_id}
    assert len(calls) == 24

    expected = [(model_id, "v1", f"T{i:02d}") for i in range(1, 13)] + [
        (model_id, "v2", f"T{i:02d}") for i in range(1, 13)
    ]
    observed = [
        (mid, request.prompt_version, request.case_id)
        for mid, request in zip(model_ids, calls, strict=True)
    ]
    assert observed == expected

    usage_path = tmp_path / "runs" / f"{run_id}.jsonl"
    output_path = tmp_path / "runs" / f"{run_id}.outputs.jsonl"
    usage = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(usage) == 24
    assert len(outputs) == 24
    assert {row["run_id"] for row in usage} == {run_id}
    assert {row["run_id"] for row in outputs} == {run_id}
    assert all(row["model_id"] == model_id for row in outputs)
    assert all(row["model_name"] == MODEL_NAME for row in outputs)
    assert all(row["task"] == "triage" for row in outputs)
    assert all(row["succeeded"] is True for row in outputs)
    assert {row["prompt_version"] for row in outputs[:12]} == {"v1"}
    assert {row["prompt_version"] for row in outputs[12:]} == {"v2"}
    assert all("analysis" not in (row["output"] or {}) for row in outputs[:12])
    assert all("analysis" in (row["output"] or {}) for row in outputs[12:])

    scores_path = tmp_path / "runs" / f"{run_id}.scores.jsonl"
    scores = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    assert len(scores) == 24 * 6
    assert {row["run_id"] for row in scores} == {run_id}
    assert {row["prompt_version"] for row in scores} == {"v1", "v2"}
    assert all(row["task"] == "triage" for row in scores)
    assert "human_review_required" not in {row["metric"] for row in scores}


class _RepairingTriageAdapter:
    provider = "ollama"

    def __init__(self, model_id: str, invalid: str, valid: str) -> None:
        self.model_id = model_id
        self._invalid = invalid
        self._valid = valid
        self.calls = 0
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        self.requests.append(request)
        text = self._invalid if self.calls == 1 else self._valid
        return CompletionResult(
            succeeded=True,
            text=text,
            error_type=None,
            records=[_stub_record(request, run_id, self.model_id)],
        )


@pytest.mark.parametrize(
    ("prompt_version", "schema", "invalid", "valid"),
    [
        ("v1", TriageOutput, '{"queue":"card_dispute"}', _triage_json(with_analysis=False)),
        (
            "v2",
            TriageOutputWithAnalysis,
            _triage_json(with_analysis=False),
            _triage_json(with_analysis=True),
        ),
    ],
)
def test_run_case_repairs_once_with_pydantic_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_version: str,
    schema: type[BaseModel],
    invalid: str,
    valid: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    model_id = Settings.from_env().models[MODEL_NAME].model_id
    adapter = _RepairingTriageAdapter(model_id, invalid, valid)
    case = load_triage_cases()[0]
    request = build_request(case, prompt_version=prompt_version, schema=schema)

    record = run_case(
        adapter,
        request,
        schema,
        run_id="triage-repair",
        max_repairs=1,
    )

    assert adapter.calls == 2
    assert record.succeeded is True
    assert record.repairs == 1
    assert record.error is None
    assert record.output is not None
    repair_text = adapter.requests[1].user_content.lower()
    assert "validation" in repair_text or "field required" in repair_text
    usage = [
        json.loads(line)
        for line in (tmp_path / "runs" / "triage-repair.jsonl").read_text().splitlines()
        if line.strip()
    ]
    outputs = [
        json.loads(line)
        for line in (tmp_path / "runs" / "triage-repair.outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(usage) == 2
    assert len(outputs) == 1
    assert outputs[0]["repairs"] == 1
    if prompt_version == "v2":
        assert "analysis" in record.output
    else:
        assert "analysis" not in record.output
