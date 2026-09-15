from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.config import Settings
from promptlab.records import OutputRecord, append_record


def complete_structured[T: BaseModel](
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[T],
    run_id: str,
    max_repairs: int = 1,
) -> OutputRecord:
    """Return a schema-validated completion with a bounded semantic repair loop.

    Transport retry remains inside the adapter.
    Schema/content repair belongs here.

    On validation failure, send the validation error text back to the model and
    instruct it to correct only what the error concerns. Do not perform more
    than max_repairs semantic repair attempts. After the bound, persist a failed
    OutputRecord instead of raising.
    """
    current = request
    last_error = "schema validation failed"
    calls = 0

    for repair_n in range(max_repairs + 1):
        result = adapter.complete(current, run_id)
        calls += 1
        if not result.succeeded:
            last_error = result.error_type or "completion failed"
            break

        parsed, error = _validate(result, schema)
        if parsed is not None:
            return _record_outcome(
                adapter,
                request,
                run_id,
                succeeded=True,
                repairs=max(0, calls - 1),
                output=parsed.model_dump(),
                error=None,
            )

        last_error = error
        if repair_n >= max_repairs:
            break
        current = _repair_request(request, error)

    return _record_outcome(
        adapter,
        request,
        run_id,
        succeeded=False,
        repairs=max(0, calls - 1),
        output=None,
        error=last_error,
    )


def _validate[T: BaseModel](
    result: CompletionResult, schema: type[T]
) -> tuple[T | None, str]:
    if result.text is None:
        return None, "empty completion"
    try:
        data = _load_json(result.text)
    except json.JSONDecodeError as exc:
        return None, f"validation failed: response is not valid JSON: {exc}"
    try:
        return schema.model_validate(data), ""
    except ValidationError as exc:
        return None, str(exc)


def _load_json(text: str) -> Any:
    """Parse a JSON object, allowing a Markdown fence or leading commentary."""
    stripped = text.strip()
    if stripped.startswith("```"):
        rest = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        fence = rest.rfind("```")
        if fence != -1:
            rest = rest[:fence]
        stripped = rest.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _repair_request(request: CompletionRequest, error: str) -> CompletionRequest:
    return request.model_copy(
        update={
            "user_content": (
                f"{request.user_content}\n\n"
                "Your previous response failed validation with the following error. "
                "Return only the corrected JSON object. Do not wrap it in Markdown fences. "
                "Do not add commentary. Do not change any field the error does not concern.\n"
                f"<error>\n{error}\n</error>"
            )
        }
    )


def _record_outcome(
    adapter: ModelAdapter,
    request: CompletionRequest,
    run_id: str,
    *,
    succeeded: bool,
    repairs: int,
    output: dict[str, Any] | None,
    error: str | None,
) -> OutputRecord:
    record = OutputRecord(
        run_id=run_id,
        task=request.task,
        case_id=request.case_id,
        model_name=_model_name(adapter.model_id),
        model_id=adapter.model_id,
        prompt_version=request.prompt_version,
        succeeded=succeeded,
        repairs=repairs,
        output=output,
        error=error,
    )
    append_record(Path("runs") / f"{run_id}.outputs.jsonl", record)
    return record


def _model_name(model_id: str) -> str:
    for name, config in Settings.from_env().models.items():
        if config.model_id == model_id:
            return name
    return model_id
