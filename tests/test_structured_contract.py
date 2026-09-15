from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.structured import complete_structured


class TinySchema(BaseModel):
    value: str


class RepairingStubAdapter:
    provider = "ollama"
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        assert run_id == "fixture-run"
        self.calls += 1
        self.requests.append(request)

        text = '{"wrong":"shape"}' if self.calls == 1 else '{"value":"fixed"}'
        return CompletionResult(
            succeeded=True,
            text=text,
            error_type=None,
            records=[],
        )


class UnrepairableStubAdapter:
    provider = "ollama"
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        self.requests.append(request)
        return CompletionResult(
            succeeded=True,
            text='{"wrong":"shape"}',
            error_type=None,
            records=[],
        )


def _request() -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id="S00",
        prompt_id="summarize",
        prompt_version="v1",
        system="",
        user_content="Summarize the supplied procedure.",
        temperature=0.0,
        max_output_tokens=128,
    )


def _outputs(run_id: str) -> list[dict[str, object]]:
    path = Path("runs") / f"{run_id}.outputs.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_complete_structured_repairs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = RepairingStubAdapter()

    result = complete_structured(
        adapter,
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )

    assert result.succeeded is True
    assert result.output == {"value": "fixed"}
    assert result.repairs == 1
    assert result.error is None
    assert adapter.calls == 2
    rows = _outputs("fixture-run")
    assert len(rows) == 1
    assert rows[0]["succeeded"] is True
    assert rows[0]["output"] == {"value": "fixed"}


class FencedStubAdapter:
    provider = "ollama"
    model_id = "fixture-model"

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        assert run_id == "fixture-run"
        return CompletionResult(
            succeeded=True,
            text='Here is the object:\n```json\n{"value":"ok"}\n```',
            error_type=None,
            records=[],
        )


def test_complete_structured_parses_fenced_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = complete_structured(
        FencedStubAdapter(),
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )
    assert result.succeeded is True
    assert result.output == {"value": "ok"}
    assert result.repairs == 0


def test_repair_request_carries_validation_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = RepairingStubAdapter()

    complete_structured(
        adapter,
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )

    repair_text = adapter.requests[1].user_content.lower()
    assert "validation" in repair_text or "field required" in repair_text
    assert "value" in repair_text


def test_complete_structured_records_failure_after_one_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = UnrepairableStubAdapter()

    result = complete_structured(
        adapter,
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )

    assert result.succeeded is False
    assert result.output is None
    assert result.repairs == 1
    assert result.error
    assert adapter.calls == 2
    rows = _outputs("fixture-run")
    assert len(rows) == 1
    assert rows[0]["succeeded"] is False
    assert rows[0]["case_id"] == "S00"
    assert rows[0]["error"] == result.error
