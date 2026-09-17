"""Final evaluation runner: 3 tasks × 2 models × 12 cases through the Day 3/4 call path."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.corpus import GoldLabel, load_cases, validate_corpus
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord, ScoreRecord, UsageRecord, append_record
from promptlab.report import write_reports
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    TaskName,
    TriageOutput,
    schema_description,
)
from promptlab.scoring import failure_scores, score_output, score_version_selection
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord
from promptlab.usage import append_record as append_usage

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MAX_OUTPUT_TOKENS = 512
TASK_CHOICES: tuple[TaskName, ...] = ("triage", "summarization", "extraction")

Kind = Literal["primary", "transport_retry", "repair", "repair_retry"]


@dataclass(frozen=True)
class TaskSpec:
    prompt_id: str
    prompt_version: str
    schema: type[BaseModel]


# Day 4 selected triage.v1: v2 added tokens/latency without changing routing.
# Extraction stays on extract.v2 except Qwen, which truncates JSON on v2.
TASK_SPECS: dict[TaskName, TaskSpec] = {
    "triage": TaskSpec("triage", "v1", TriageOutput),
    "summarization": TaskSpec("summarize", "v1", SummarizationOutput),
    "extraction": TaskSpec("extract", "v2", PolicyExtraction),
}

PROMPT_VERSION_BY_MODEL: dict[tuple[TaskName, str], str] = {
    ("extraction", "qwen"): "v3",
}


def task_spec_for(
    task: TaskName,
    model_name: str,
    prompt_version: str | None = None,
) -> TaskSpec:
    spec = TASK_SPECS[task]
    if prompt_version is not None:
        return TaskSpec(spec.prompt_id, prompt_version, spec.schema)
    version = PROMPT_VERSION_BY_MODEL.get((task, model_name), spec.prompt_version)
    return TaskSpec(spec.prompt_id, version, spec.schema)


class RecordingAdapter:
    """Persist every adapter attempt, including semantic repairs and transport retries."""

    def __init__(self, inner: ModelAdapter) -> None:
        self.provider = inner.provider
        self.model_id = inner.model_id
        self._inner = inner
        self._complete_n = 0
        self._pending: list[tuple[CallRecord, int]] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self._complete_n += 1
        result = self._inner.complete(request, run_id)
        for record in result.records:
            append_usage(record, run_id)
            self._pending.append((record, self._complete_n))
        return result

    def take_case_attempts(self) -> list[tuple[CallRecord, int]]:
        rows = self._pending
        self._pending = []
        self._complete_n = 0
        return rows


def _parser(model_names: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local two-model prompt comparison")
    parser.add_argument("--run-id", help="Stable identifier for this run")
    parser.add_argument("--task", choices=TASK_CHOICES)
    parser.add_argument("--model", choices=model_names)
    parser.add_argument(
        "--prompt-version",
        help=(
            "Override prompt version for the selected task on every selected "
            "model (requires --task). Default extraction is v2 on mistral and "
            "v3 on qwen"
        ),
    )
    parser.add_argument("--limit", type=int, help="Limit cases per task for a smoke run")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and corpus without calling Ollama",
    )
    return parser


def _instance_rules(task: TaskName, schema: type[BaseModel]) -> str:
    rules = (
        f"Return only a JSON instance of {schema.__name__}. "
        "Do not return JSON Schema. Do not include $defs, properties, type, "
        "required, title, or additionalProperties. Do not use Markdown fences. "
        "Do not add keys that are not in the schema."
    )
    if task == "triage":
        return rules
    return (
        rules
        + " value must be a string, a list of strings, or null — never an object. "
        'An absent field is {"value": null, "status": "absent"}. '
        "A present citation must be the full heading line copied from the source "
        '(for example "1. Document Control"), not a number, label, or invented heading.'
    )


def build_request(
    *,
    task: TaskName,
    case_id: str,
    document_text: str,
    spec: TaskSpec,
    temperature: float,
) -> CompletionRequest:
    template = load(spec.prompt_id, spec.prompt_version)
    variables: dict[str, str] = {}
    if (
        "{schema_description}" in template.user_template
        or "{schema_description}" in template.system
    ):
        variables["schema_description"] = schema_description(spec.schema)
    system = template.system
    if "{schema_description}" in system:
        system = system.replace("{schema_description}", variables["schema_description"])
    rules = _instance_rules(task, spec.schema)
    return CompletionRequest(
        task=task,
        case_id=case_id,
        prompt_id=spec.prompt_id,
        prompt_version=spec.prompt_version,
        system=f"{system}\n\n{rules}" if system else rules,
        user_content=render_user(template, variables, untrusted=document_text),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _write_run_timing(run_id: str, started_at: datetime, ended_at: datetime) -> None:
    """Persist one wall-clock span for the whole run, retries included."""
    elapsed_ms = max(0, int((ended_at - started_at).total_seconds() * 1000))
    path = Path("runs") / f"{run_id}.timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_ms": elapsed_ms,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"run_id={run_id} ended_at={ended_at.isoformat()} elapsed_ms={elapsed_ms}")


def _usage_from_call(
    record: CallRecord,
    *,
    model_name: str,
    is_repair: bool,
) -> UsageRecord:
    kind: Kind
    if is_repair:
        kind = "repair_retry" if record.attempt > 1 else "repair"
    else:
        kind = "transport_retry" if record.attempt > 1 else "primary"
    status: Literal["success", "schema_invalid", "transport_error"] = (
        "success" if record.error_type is None else "transport_error"
    )
    return UsageRecord(
        run_id=record.run_id,
        task=record.task,
        case_id=record.case_id,
        model_name=model_name,
        model_id=record.model_id,
        prompt_id=record.prompt_id,
        prompt_version=record.prompt_version,
        attempt=record.attempt,
        kind=kind,
        status=status,
        prompt_tokens=record.input_tokens,
        completion_tokens=record.output_tokens,
        latency_ms=float(record.latency_ms),
        cost_usd=Decimal(str(record.cost_usd)),
        error=record.error_type,
    )


def main(argv: Sequence[str] | None = None) -> str | None:
    settings = Settings.from_env()
    args = _parser(tuple(settings.models)).parse_args(
        list(argv) if argv is not None else None
    )
    counts = validate_corpus()
    if args.validate_only:
        print("Corpus valid: " + ", ".join(f"{task}={count}" for task, count in counts.items()))
        return None

    if not isinstance(args.run_id, str) or not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise SystemExit("--run-id is required and must use letters, numbers, '.', '_' or '-'")
    run_id = args.run_id
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    selected_tasks: list[TaskName] = (
        [cast(TaskName, args.task)] if args.task else list(TASK_CHOICES)
    )
    selected_models = [args.model] if args.model else list(settings.models)
    if args.prompt_version and args.task is None:
        raise SystemExit("--prompt-version requires --task")
    for task in selected_tasks:
        for model_name in selected_models:
            spec = task_spec_for(task, model_name, args.prompt_version)
            load(spec.prompt_id, spec.prompt_version)

    usage_path = Path("runs") / f"{run_id}.jsonl"
    if usage_path.exists():
        raise SystemExit(f"Run already exists: {usage_path}")

    started_at = datetime.now(UTC)
    print(f"run_id={run_id} started_at={started_at.isoformat()}")
    try:
        all_usage: list[UsageRecord] = []
        all_outputs: list[OutputRecord] = []
        all_scores: list[ScoreRecord] = []
        validated_by_task_model: dict[tuple[TaskName, str], dict[str, BaseModel]] = defaultdict(
            dict
        )
        labels_by_task: dict[TaskName, list[GoldLabel]] = {}
        scores_path = Path("runs") / f"{run_id}.scores.jsonl"
        adapters = {
            name: RecordingAdapter(OllamaAdapter(model_id=settings.models[name].model_id))
            for name in selected_models
        }

        for task in selected_tasks:
            pairs = load_cases(task)
            if args.limit is not None:
                pairs = pairs[: args.limit]
            labels_by_task[task] = [gold for _case, gold in pairs]
            for model_name in selected_models:
                spec = task_spec_for(task, model_name, args.prompt_version)
                adapter = adapters[model_name]
                for case, gold in pairs:
                    request = build_request(
                        task=task,
                        case_id=case.id,
                        document_text=case.document_text,
                        spec=spec,
                        temperature=settings.temperature,
                    )
                    output_record = complete_structured(
                        adapter,
                        request,
                        spec.schema,
                        run_id,
                        max_repairs=settings.max_schema_repairs,
                    )
                    for call, complete_n in adapter.take_case_attempts():
                        all_usage.append(
                            _usage_from_call(
                                call,
                                model_name=model_name,
                                is_repair=complete_n > 1,
                            )
                        )
                    all_outputs.append(output_record)
                    if output_record.succeeded and output_record.output is not None:
                        parsed = spec.schema.model_validate(output_record.output)
                        validated_by_task_model[(task, model_name)][case.id] = parsed
                        case_scores = score_output(
                            run_id=run_id,
                            task=task,
                            case_id=case.id,
                            model_name=model_name,
                            model_id=adapter.model_id,
                            prompt_id=spec.prompt_id,
                            prompt_version=spec.prompt_version,
                            output=parsed,
                            gold=gold,
                            source=case.document_text,
                        )
                    else:
                        case_scores = failure_scores(
                            run_id=run_id,
                            task=task,
                            case_id=case.id,
                            model_name=model_name,
                            model_id=adapter.model_id,
                            prompt_id=spec.prompt_id,
                            prompt_version=spec.prompt_version,
                            gold=gold,
                        )
                    for score in case_scores:
                        append_record(scores_path, score)
                        all_scores.append(score)
                    status = "ok" if output_record.succeeded else "failed"
                    print(f"{task:13} {model_name:8} {case.id:5} {status}")

        for task in selected_tasks:
            if task == "triage":
                continue
            for model_name in selected_models:
                spec = task_spec_for(task, model_name, args.prompt_version)
                for record in score_version_selection(
                    run_id=run_id,
                    task=task,
                    model_name=model_name,
                    model_id=settings.models[model_name].model_id,
                    prompt_id=spec.prompt_id,
                    prompt_version=spec.prompt_version,
                    labels=labels_by_task[task],
                    outputs=validated_by_task_model[(task, model_name)],
                ):
                    append_record(scores_path, record)
                    all_scores.append(record)

        report_path = Path("reports") / "comparison.md"
        decision_path = Path("docs") / "model-decision.md"
        write_reports(
            run_id=run_id,
            models=selected_models,
            usage=all_usage,
            outputs=all_outputs,
            scores=all_scores,
            report_path=report_path,
            decision_path=decision_path,
        )
        total_cost = sum((row.cost_usd for row in all_usage), Decimal("0"))
        print(f"Report: {report_path}")
        print(f"Recorded provider cost: ${total_cost}")
        return run_id
    finally:
        _write_run_timing(run_id, started_at, datetime.now(UTC))


if __name__ == "__main__":
    print(main())
