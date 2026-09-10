"""Day 1 local Ollama extraction run."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, append_record, compute_cost

SELECTED_CASE_IDS = ("E12", "E07", "E11")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 512
TRUNCATION_NUM_PREDICT = 8


def load_selected_cases() -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    path = PROJECT_ROOT / "cases" / "extraction.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw["id"])
        by_id[case_id] = {
            "id": case_id,
            "task": str(raw["task"]),
            "source": str(raw["source"]),
        }
    return [by_id[case_id] for case_id in SELECTED_CASE_IDS]


def generate(
    settings: Settings, model_id: str, prompt: str, num_predict: int
) -> tuple[Any, int]:
    started = time.perf_counter()
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": num_predict,
            },
        },
        timeout=180.0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    return response.json(), latency_ms


def build_record(
    *,
    model_id: str,
    case_id: str,
    run_id: str,
    payload: Any,
    latency_ms: int,
    num_predict: int,
    error_type: str | None,
) -> CallRecord:
    input_tokens = int(payload.get("prompt_eval_count") or 0)
    output_tokens = int(payload.get("eval_count") or 0)
    done_reason = payload.get("done_reason")
    text = payload.get("response")
    return CallRecord(
        record_id=str(uuid.uuid4()),
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="extraction",
        case_id=case_id,
        prompt_id="baseline",
        prompt_version="v0",
        attempt=1,
        temperature=TEMPERATURE,
        max_output_tokens=num_predict,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        latency_ms=latency_ms,
        cost_usd=compute_cost(model_id, input_tokens, output_tokens),
        stop_reason=None if done_reason is None else str(done_reason),
        error_type=error_type,
        response_text=None if text is None else str(text),
    )


def main() -> None:
    settings = Settings.from_env()
    model = settings.models["mistral"]
    template = (PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md").read_text(encoding="utf-8")
    run_id = str(uuid.uuid4())
    cases = load_selected_cases()

    e11 = next(case for case in cases if case["id"] == "E11")
    truncation_payload, truncation_latency_ms = generate(
        settings,
        model.model_id,
        template.replace("{document_text}", e11["source"]),
        TRUNCATION_NUM_PREDICT,
    )
    truncation_error = (
        "TruncatedResponseError" if truncation_payload.get("done_reason") == "length" else None
    )
    append_record(
        build_record(
            model_id=model.model_id,
            case_id=e11["id"],
            run_id=f"{run_id}-truncation",
            payload=truncation_payload,
            latency_ms=truncation_latency_ms,
            num_predict=TRUNCATION_NUM_PREDICT,
            error_type=truncation_error,
        ),
        f"{run_id}-truncation",
    )

    for case in cases:
        payload, latency_ms = generate(
            settings,
            model.model_id,
            template.replace("{document_text}", case["source"]),
            MAX_OUTPUT_TOKENS,
        )
        append_record(
            build_record(
                model_id=model.model_id,
                case_id=case["id"],
                run_id=run_id,
                payload=payload,
                latency_ms=latency_ms,
                num_predict=MAX_OUTPUT_TOKENS,
                error_type=None,
            ),
            run_id,
        )


if __name__ == "__main__":
    main()
