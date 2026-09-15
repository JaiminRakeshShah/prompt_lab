"""Day 3 structured-output run: summarization v1 and extraction v2."""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.records import OutputRecord
from promptlab.schemas import PolicyExtraction, SummarizationOutput, TaskName, render_prompt
from promptlab.structured import complete_structured
from promptlab.usage import append_record as append_usage

MODEL_NAME = "mistral"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 512


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


def load_cases(filename: str) -> list[dict[str, str]]:
    path = PROJECT_ROOT / "cases" / filename
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


def load_prompt(filename: str) -> str:
    return (PROJECT_ROOT / "src" / "prompts" / filename).read_text(encoding="utf-8")


def build_request(
    case: dict[str, str],
    *,
    task: TaskName,
    prompt_id: str,
    prompt_version: str,
    template: str,
    schema: type[BaseModel],
) -> CompletionRequest:
    return CompletionRequest(
        task=task,
        case_id=case["id"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        system=(
            f"Return only a JSON instance of {schema.__name__}. "
            "Do not return JSON Schema. Do not include $defs, properties, type, "
            "required, title, or additionalProperties. Do not use Markdown fences."
        ),
        user_content=render_prompt(template, schema, case["source"]),
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


def _run_task(
    adapter: ModelAdapter,
    *,
    run_id: str,
    max_repairs: int,
    task: TaskName,
    cases_file: str,
    prompt_file: str,
    prompt_id: str,
    prompt_version: str,
    schema: type[BaseModel],
) -> None:
    template = load_prompt(prompt_file)
    for case in load_cases(cases_file):
        request = build_request(
            case,
            task=task,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            template=template,
            schema=schema,
        )
        record = run_case(
            adapter,
            request,
            schema,
            run_id=run_id,
            max_repairs=max_repairs,
        )
        status = "ok" if record.succeeded else "fail"
        print(f"{request.case_id} {status} repairs={record.repairs}")


def main() -> str:
    settings = Settings.from_env()
    model = settings.models[MODEL_NAME]
    adapter = OllamaAdapter(model_id=model.model_id)
    run_id = str(uuid.uuid4())
    print(f"run_id={run_id} model={model.model_id} temperature={TEMPERATURE}")

    _run_task(
        adapter,
        run_id=run_id,
        max_repairs=settings.max_schema_repairs,
        task="summarization",
        cases_file="summarization.jsonl",
        prompt_file="summarize.v1.md",
        prompt_id="summarize",
        prompt_version="v1",
        schema=SummarizationOutput,
    )
    _run_task(
        adapter,
        run_id=run_id,
        max_repairs=settings.max_schema_repairs,
        task="extraction",
        cases_file="extraction.jsonl",
        prompt_file="extract.v2.md",
        prompt_id="extract",
        prompt_version="v2",
        schema=PolicyExtraction,
    )
    return run_id


if __name__ == "__main__":
    print(main())
