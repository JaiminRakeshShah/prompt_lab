"""Day 2 local Ollama summarization run across Mistral and Qwen."""

from __future__ import annotations

import json
import uuid

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import append_record

MODEL_NAMES = ("mistral", "qwen")
MAX_OUTPUT_TOKENS = 512
PROMPT_ID = "baseline"
PROMPT_VERSION = "v0"


def load_summarization_cases() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "cases" / "summarization.jsonl"
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


def load_prompt_parts() -> tuple[str, str]:
    template = (PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md").read_text(encoding="utf-8")
    system, separator, document_block = template.partition("<document>")
    if separator == "":
        raise ValueError("baseline prompt is missing a <document> section")
    return system.strip(), "<document>" + document_block


def build_request(
    case: dict[str, str],
    *,
    system: str,
    document_template: str,
    temperature: float,
) -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id=case["id"],
        prompt_id=PROMPT_ID,
        prompt_version=PROMPT_VERSION,
        system=system,
        user_content=document_template.replace("{document_text}", case["source"]),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def main() -> str:
    settings = Settings.from_env()
    system, document_template = load_prompt_parts()
    cases = load_summarization_cases()
    run_id = str(uuid.uuid4())

    for name in MODEL_NAMES:
        adapter = OllamaAdapter(model_id=settings.models[name].model_id)
        for case in cases:
            result = adapter.complete(
                build_request(
                    case,
                    system=system,
                    document_template=document_template,
                    temperature=settings.temperature,
                ),
                run_id,
            )
            for record in result.records:
                append_record(record, run_id)

    return run_id


if __name__ == "__main__":
    print(main())
