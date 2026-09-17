from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.run import (
    MAX_OUTPUT_TOKENS,
    PROMPT_VERSION_BY_MODEL,
    TASK_SPECS,
    main,
    task_spec_for,
)
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


def _triage_json() -> str:
    return json.dumps(
        {
            "queue": "card_dispute",
            "escalation_required": False,
            "confidence": 0.8,
            "rationale": "Recognized duplicate charge.",
            "draft_reply": "We will review the duplicate charge.",
            "human_review_required": True,
            "customer_outcome": None,
        }
    )


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


def _text_for(request: CompletionRequest) -> str:
    if request.task == "triage":
        return _triage_json()
    if request.task == "summarization":
        return _summarization_json()
    return _extraction_json()


def _assert_run_timing(tmp_path: Path, run_id: str) -> None:
    payload = json.loads((tmp_path / "runs" / f"{run_id}.timing.json").read_text())
    assert isinstance(payload, dict)
    started = datetime.fromisoformat(str(payload["started_at"]))
    ended = datetime.fromisoformat(str(payload["ended_at"]))
    assert payload["run_id"] == run_id
    assert started.tzinfo is not None
    assert ended.tzinfo is not None
    assert ended >= started
    elapsed_ms = payload["elapsed_ms"]
    assert isinstance(elapsed_ms, int)
    assert elapsed_ms == max(0, int((ended - started).total_seconds() * 1000))
    assert elapsed_ms >= 0


def test_run_module_uses_registry_and_adapter_path() -> None:
    import promptlab.run as run

    source = inspect.getsource(run)
    assert "httpx" not in source
    assert "/api/generate" not in source
    assert "complete_structured" in source
    assert "OllamaAdapter" in source
    assert "render_user" in source
    assert "load(" in source
    assert "ModelAdapter" in source


def test_day5_prompt_versions_are_fixed() -> None:
    assert TASK_SPECS["summarization"].prompt_id == "summarize"
    assert TASK_SPECS["summarization"].prompt_version == "v1"
    assert TASK_SPECS["extraction"].prompt_id == "extract"
    assert TASK_SPECS["extraction"].prompt_version == "v2"
    assert TASK_SPECS["triage"].prompt_id == "triage"
    assert TASK_SPECS["triage"].prompt_version == "v1"
    assert PROMPT_VERSION_BY_MODEL == {("extraction", "qwen"): "v3"}
    assert task_spec_for("extraction", "mistral").prompt_version == "v2"
    assert task_spec_for("extraction", "qwen").prompt_version == "v3"
    assert task_spec_for("extraction", "qwen", "v2").prompt_version == "v2"


def test_default_run_is_seventy_two_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[CompletionRequest] = []
    model_ids: list[str] = []
    name_by_id = {cfg.model_id: name for name, cfg in Settings.from_env().models.items()}

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append(request)
        model_ids.append(self.model_id)
        spec = task_spec_for(request.task, name_by_id[self.model_id])
        assert request.prompt_id == spec.prompt_id
        assert request.prompt_version == spec.prompt_version
        assert request.temperature == 0.0
        assert request.max_output_tokens == MAX_OUTPUT_TOKENS
        assert "JSON instance" in request.system
        assert "{document_text}" not in request.user_content
        assert "{schema_description}" not in request.user_content
        assert request.case_id not in request.system
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(
            succeeded=True,
            text=_text_for(request),
            error_type=None,
            records=[record],
        )

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main(["--run-id", "local-comparison-01"])
    assert run_id == "local-comparison-01"

    settings = Settings.from_env()
    expected: list[tuple[str, str, str]] = []
    for task in ("triage", "summarization", "extraction"):
        prefix = {"triage": "T", "summarization": "S", "extraction": "E"}[task]
        for name in settings.models:
            for i in range(1, 13):
                expected.append((settings.models[name].model_id, task, f"{prefix}{i:02d}"))
    observed = [
        (model_id, request.task, request.case_id)
        for model_id, request in zip(model_ids, calls, strict=True)
    ]
    assert observed == expected
    assert len(calls) == 72

    usage = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    outputs = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    scores = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(usage) == 72
    assert len(outputs) == 72
    assert {row["run_id"] for row in usage} == {run_id}
    assert {row["run_id"] for row in outputs} == {run_id}
    assert all(row["succeeded"] is True for row in outputs)
    assert all(row["cost_usd"] == 0.0 for row in usage)
    assert all(row["provider"] == "ollama" for row in usage)
    assert {row["prompt_version"] for row in outputs if row["task"] == "triage"} == {"v1"}
    assert {row["prompt_version"] for row in outputs if row["task"] == "summarization"} == {"v1"}
    assert {
        row["prompt_version"]
        for row in outputs
        if row["task"] == "extraction" and row["model_name"] == "mistral"
    } == {"v2"}
    assert {
        row["prompt_version"]
        for row in outputs
        if row["task"] == "extraction" and row["model_name"] == "qwen"
    } == {"v3"}
    triage_outputs = [row for row in outputs if row["task"] == "triage"]
    assert all("analysis" not in (row["output"] or {}) for row in triage_outputs)
    assert scores
    assert {row["run_id"] for row in scores} == {run_id}
    call_keys = {
        (
            row["run_id"],
            row["case_id"],
            row["task"],
            row["model_id"],
            row["prompt_id"],
            row["prompt_version"],
        )
        for row in usage
    }
    for row in scores:
        assert row["model_id"]
        assert row["prompt_id"]
        if str(row["case_id"]).startswith("version:"):
            continue
        key = (
            row["run_id"],
            row["case_id"],
            row["task"],
            row["model_id"],
            row["prompt_id"],
            row["prompt_version"],
        )
        assert key in call_keys
    assert any(row["metric"] == "version_selection_accuracy" for row in scores)
    assert (tmp_path / "reports" / "comparison.md").exists()
    assert (tmp_path / "docs" / "model-decision.md").exists()
    _assert_run_timing(tmp_path, run_id)


def test_default_extraction_uses_v2_for_mistral_and_v3_for_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    versions: list[tuple[str, str]] = []
    name_by_id = {cfg.model_id: name for name, cfg in Settings.from_env().models.items()}

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        versions.append((name_by_id[self.model_id], request.prompt_version))
        assert request.prompt_id == "extract"
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(
            succeeded=True,
            text=_extraction_json(),
            error_type=None,
            records=[record],
        )

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main(["--run-id", "mixed-extract", "--task", "extraction", "--limit", "1"])
    assert run_id == "mixed-extract"
    assert versions == [("mistral", "v2"), ("qwen", "v3")]
    outputs = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    usage = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    scores = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pairs = {(row["model_name"], row["prompt_version"]) for row in outputs}
    assert pairs == {("mistral", "v2"), ("qwen", "v3")}
    id_to_name = {cfg.model_id: name for name, cfg in Settings.from_env().models.items()}
    assert {(id_to_name[row["model_id"]], row["prompt_version"]) for row in usage} == pairs
    assert {(row["model_name"], row["prompt_version"]) for row in scores} == pairs
    assert {row["run_id"] for row in usage} | {row["run_id"] for row in scores} == {run_id}
    report = (tmp_path / "reports" / "comparison.md").read_text(encoding="utf-8")
    assert f"Run ID: `{run_id}`" in report
    assert "| Mistral | extract.v2 |" in report
    assert "| Qwen | extract.v3 |" in report


def test_prompt_version_override_records_adapted_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    versions: list[str] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        versions.append(request.prompt_version)
        assert request.prompt_id == "extract"
        assert request.prompt_version == "v3"
        assert "{schema_description}" not in request.system
        assert "{schema_description}" not in request.user_content
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(
            succeeded=True,
            text=_extraction_json(),
            error_type=None,
            records=[record],
        )

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main(
        [
            "--run-id",
            "qwen-extract-v3",
            "--task",
            "extraction",
            "--model",
            "qwen",
            "--prompt-version",
            "v3",
            "--limit",
            "1",
        ]
    )
    assert run_id == "qwen-extract-v3"
    assert versions == ["v3"]
    outputs = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert outputs[0]["prompt_version"] == "v3"
    scores = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert scores
    assert {row["prompt_version"] for row in scores} == {"v3"}
    report = (tmp_path / "reports" / "comparison.md").read_text(encoding="utf-8")
    assert "| Qwen | extract.v3 |" in report


def test_task_and_model_filters_reduce_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str, str]] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append((self.model_id, request.task, request.case_id))
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(
            succeeded=True,
            text=_text_for(request),
            error_type=None,
            records=[record],
        )

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main(["--run-id", "triage-mistral", "--task", "triage", "--model", "mistral"])
    settings = Settings.from_env()
    mistral_id = settings.models["mistral"].model_id
    assert run_id == "triage-mistral"
    assert len(calls) == 12
    assert [row[0] for row in calls] == [mistral_id] * 12
    assert [row[1] for row in calls] == ["triage"] * 12
    assert [row[2] for row in calls] == [f"T{i:02d}" for i in range(1, 13)]


def test_repair_adds_extra_call_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls["n"] += 1
        text = '{"queue":"card_dispute"}' if calls["n"] == 1 else _triage_json()
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(succeeded=True, text=text, error_type=None, records=[record])

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_id = main(
        [
            "--run-id",
            "repair-run",
            "--task",
            "triage",
            "--model",
            "mistral",
            "--limit",
            "1",
        ]
    )
    assert run_id == "repair-run"
    usage = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    outputs = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run_id}.outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert calls["n"] == 2
    assert len(usage) == 2
    assert len(outputs) == 1
    assert outputs[0]["succeeded"] is True
    assert outputs[0]["repairs"] == 1
    _assert_run_timing(tmp_path, run_id)


def test_timing_is_written_if_the_run_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        raise RuntimeError("adapter failed")

    monkeypatch.setattr(OllamaAdapter, "complete", fail_complete)
    with pytest.raises(RuntimeError, match="adapter failed"):
        main(
            [
                "--run-id",
                "crash-run",
                "--task",
                "triage",
                "--model",
                "mistral",
                "--limit",
                "1",
            ]
        )
    _assert_run_timing(tmp_path, "crash-run")


def test_run_id_is_required() -> None:
    with pytest.raises(SystemExit, match="--run-id"):
        main([])


def test_model_flag_uses_configured_logical_names_not_identifiers() -> None:
    settings = Settings.from_env()
    mistral = settings.models["mistral"]
    assert mistral.model_id != "mistral"
    with pytest.raises(SystemExit):
        main(["--run-id", "bad-model", "--model", mistral.model_id])


def test_validate_only_does_not_call_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        raise AssertionError("validate-only must not call Ollama")

    monkeypatch.setattr(OllamaAdapter, "complete", fail_complete)
    assert main(["--validate-only"]) is None
    assert not (tmp_path / "runs").exists()
