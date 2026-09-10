from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from promptlab.config import Settings
from promptlab.day1 import (
    MAX_OUTPUT_TOKENS,
    TRUNCATION_NUM_PREDICT,
    build_record,
    load_selected_cases,
    main,
)


def test_load_selected_cases_returns_short_mid_long() -> None:
    cases = load_selected_cases()
    assert [case["id"] for case in cases] == ["E12", "E07", "E11"]
    assert all(case["source"] for case in cases)


def test_build_record_maps_ollama_payload() -> None:
    model_id = Settings.from_env().models["mistral"].model_id
    record = build_record(
        model_id=model_id,
        case_id="E12",
        run_id="unit-test",
        payload={
            "prompt_eval_count": 12,
            "eval_count": 4,
            "done_reason": "stop",
            "response": "hello",
        },
        latency_ms=100,
        num_predict=512,
        error_type=None,
    )

    assert record.input_tokens == 12
    assert record.output_tokens == 4
    assert record.stop_reason == "stop"
    assert record.response_text == "hello"
    assert record.cost_usd == 0.0
    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset() is not None
    assert uuid.UUID(record.record_id).version == 4


def test_main_keeps_truncation_out_of_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    num_predicts: list[int] = []
    prompts: list[str] = []

    def fake_generate(
        settings: Settings, model_id: str, prompt: str, num_predict: int
    ) -> tuple[dict[str, object], int]:
        num_predicts.append(num_predict)
        prompts.append(prompt)
        if num_predict == TRUNCATION_NUM_PREDICT:
            return (
                {
                    "prompt_eval_count": 10,
                    "eval_count": 8,
                    "done_reason": "length",
                    "response": "cut",
                },
                50,
            )
        return (
            {
                "prompt_eval_count": 10,
                "eval_count": 20,
                "done_reason": "stop",
                "response": "ok",
            },
            100,
        )

    monkeypatch.setattr("promptlab.day1.generate", fake_generate)
    main()

    assert num_predicts[0] == TRUNCATION_NUM_PREDICT
    assert num_predicts[1:] == [MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS]
    assert all("{document_text}" not in prompt for prompt in prompts)

    run_files = list((tmp_path / "runs").glob("*.jsonl"))
    truncation_path = next(path for path in run_files if path.stem.endswith("-truncation"))
    success_path = next(path for path in run_files if not path.stem.endswith("-truncation"))

    truncation = json.loads(truncation_path.read_text(encoding="utf-8").splitlines()[0])
    assert truncation["case_id"] == "E11"
    assert truncation["stop_reason"] == "length"
    assert truncation["error_type"] == "TruncatedResponseError"
    assert truncation["max_output_tokens"] == TRUNCATION_NUM_PREDICT

    successes = [
        json.loads(line) for line in success_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["case_id"] for record in successes] == ["E12", "E07", "E11"]
    assert all(record["error_type"] is None for record in successes)
    assert all(record["stop_reason"] == "stop" for record in successes)
    assert all(record["max_output_tokens"] == MAX_OUTPUT_TOKENS for record in successes)
    assert len(successes) == 3
