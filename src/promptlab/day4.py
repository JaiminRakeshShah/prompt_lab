"""Day 4 triage comparison: prompt v1 vs v2 on one Mistral run."""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
from promptlab.scoring import load_triage_gold, write_scores
from promptlab.structured import complete_structured
from promptlab.usage import append_record as append_usage

MODEL_NAME = "mistral"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 512
PROMPT_ID = "triage"
PROMPT_VERSIONS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("v1", TriageOutput),
    ("v2", TriageOutputWithAnalysis),
)


class RecordingAdapter:
    """Persist every adapter attempt, including the one semantic repair."""

    def __init__(self, inner: ModelAdapter) -> None:
        self.provider = inner.provider
        self.model_id = inner.model_id
        self._inner = inner

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        result = self._inner.complete(request, run_id)
        for record in result.records:
            append_usage(record, run_id)
        return result


def load_triage_cases() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "cases" / "triage.jsonl"
    cases: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            {
                "id": str(raw["id"]),
                "task": str(raw["task"]),
                "source": str(raw["source"]),
            }
        )
    return cases


def _instance_rules(schema: type[BaseModel]) -> str:
    """Same JSON-instance rules Day 3 added so Pydantic validation can succeed."""
    return (
        f"Return only a JSON instance of {schema.__name__}. "
        "Do not return JSON Schema. Do not include $defs, properties, type, "
        "required, title, or additionalProperties. Do not use Markdown fences. "
        "Do not add keys that are not in the schema."
    )


def build_request(
    case: dict[str, str],
    *,
    prompt_version: str,
    schema: type[BaseModel],
) -> CompletionRequest:
    template = load(PROMPT_ID, prompt_version)
    return CompletionRequest(
        task="triage",
        case_id=case["id"],
        prompt_id=PROMPT_ID,
        prompt_version=prompt_version,
        system=f"{template.system}\n\n{_instance_rules(schema)}",
        user_content=render_user(template, variables={}, untrusted=case["source"]),
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def run_case(
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[BaseModel],
    *,
    run_id: str,
    max_repairs: int,
) -> OutputRecord:
    return complete_structured(
        RecordingAdapter(adapter),
        request,
        schema,
        run_id,
        max_repairs=max_repairs,
    )


def main() -> str:
    settings = Settings.from_env()
    model = settings.models[MODEL_NAME]
    adapter = OllamaAdapter(model_id=model.model_id)
    run_id = str(uuid.uuid4())
    cases = load_triage_cases()
    gold = load_triage_gold()
    print(f"run_id={run_id} model={model.model_id} temperature={TEMPERATURE}")

    for prompt_version, schema in PROMPT_VERSIONS:
        for case in cases:
            request = build_request(case, prompt_version=prompt_version, schema=schema)
            record = run_case(
                adapter,
                request,
                schema,
                run_id=run_id,
                max_repairs=settings.max_schema_repairs,
            )
            if record.succeeded and record.output is not None:
                parsed = schema.model_validate(record.output)
                if isinstance(parsed, TriageOutput):
                    write_scores(
                        run_id=record.run_id,
                        case_id=record.case_id,
                        model_name=record.model_name,
                        prompt_version=record.prompt_version,
                        output=parsed,
                        gold=gold[case["id"]],
                    )
            status = "ok" if record.succeeded else "fail"
            print(f"{prompt_version} {request.case_id} {status} repairs={record.repairs}")

    return run_id


if __name__ == "__main__":
    print(main())
