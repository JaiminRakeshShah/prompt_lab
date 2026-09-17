"""Reporting for the Week 2 model-comparison lab.

The reporting layer consumes the existing UsageRecord, OutputRecord, and
ScoreRecord objects.  It does not rescore model output and it does not call an
LLM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import ValidationError

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.schemas import TriageOutput
from promptlab.scoring import score_human_boundary_compliance

_ConfigKey = tuple[str, str, str, str]  # task, model_name, prompt_id, prompt_version
_RETRY_KINDS = frozenset({"transport_retry", "repair_retry"})

_NATIVE_PROMPT: dict[str, tuple[str, str]] = {
    "triage": ("triage", "v1"),
    "summarization": ("summarize", "v1"),
    "extraction": ("extract", "v2"),
}

_TRIAGE_QUALITY: tuple[tuple[str, str], ...] = (
    ("queue", "routing"),
    ("escalation", "escalation"),
    ("missed_escalation", "missed escalations"),
    ("unnecessary_escalation", "unnecessary escalations"),
    ("human_boundary_compliance", "human-boundary"),
    ("pii_leakage", "PII leakage"),
)


def _key(record: Any) -> _ConfigKey:
    prompt_id = str(getattr(record, "prompt_id", "") or "")
    return (
        str(record.task),
        str(record.model_name),
        prompt_id,
        str(record.prompt_version),
    )


def _for_runs(records: Sequence[Any], run_ids: Sequence[str]) -> list[Any]:
    allowed = set(run_ids)
    return [record for record in records if str(record.run_id) in allowed]


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _aggregate_scores(
    records: Sequence[ScoreRecord],
) -> dict[str, tuple[int, int, bool | None]]:
    """Aggregate compatible score counts without averaging percentages."""

    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.metric)].append(record)

    result: dict[str, tuple[int, int, bool | None]] = {}

    for metric, rows in sorted(grouped.items()):
        numerator = sum(int(row.numerator) for row in rows)
        denominator = sum(int(row.denominator) for row in rows)

        directions = {
            bool(value)
            for value in (getattr(row, "lower_is_better", None) for row in rows)
            if value is not None
        }
        lower_is_better: bool | None = (
            next(iter(directions)) if len(directions) == 1 else None
        )

        result[metric] = (numerator, denominator, lower_is_better)

    return result


def _ratio(numerator: int, denominator: int, *, invert: bool = False) -> str:
    if invert:
        numerator = max(0, denominator - numerator)
    return f"{numerator}/{denominator}"


def _quality_text(
    task: str,
    scores: Sequence[ScoreRecord],
    outputs: Sequence[OutputRecord],
) -> str:
    metrics = _aggregate_scores(scores)
    valid, _repairs, _failures = _output_summary(outputs)
    lines: list[str] = []
    if task == "triage":
        for metric, label in _TRIAGE_QUALITY:
            if metric not in metrics:
                continue
            numerator, denominator, lower_is_better = metrics[metric]
            suffix = " ↓" if lower_is_better else ""
            lines.append(f"{label}: {numerator}/{denominator}{suffix}")
        return "<br>".join(lines) if lines else "—"

    lines.append(f"valid: {valid}")
    recall = metrics.get("required_evidence_recall")
    if recall is not None:
        numerator, denominator, _ = recall
        lines.append(f"missed values: {_ratio(numerator, denominator, invert=True)} ↓")
    unsupported = metrics.get("unsupported_field_avoidance")
    if unsupported is not None:
        numerator, denominator, _ = unsupported
        lines.append(
            f"unsupported values: {_ratio(numerator, denominator, invert=True)} ↓"
        )
    return "<br>".join(lines) if lines else "—"


def _prompt_label(task: str, model_name: str, prompt_id: str, prompt_version: str) -> str:
    label = f"{prompt_id}.{prompt_version}"
    native = _NATIVE_PROMPT.get(task)
    if native == (prompt_id, prompt_version) and model_name != "mistral":
        return f"{label} transfer"
    return label


def _human_boundary_verification(outputs: Sequence[OutputRecord]) -> list[str]:
    """Re-run Day 4 human-boundary checks on stored triage drafts. Does not call a model."""

    by_model: dict[str, list[tuple[str, bool, str | None]]] = defaultdict(list)
    for row in outputs:
        if str(row.task) != "triage" or row.output is None:
            continue
        try:
            parsed = TriageOutput.model_validate(row.output)
        except ValidationError:
            continue
        result = score_human_boundary_compliance(parsed, parsed)[0]
        by_model[str(row.model_name)].append(
            (str(row.case_id), bool(result.passed), result.detail)
        )
    if not by_model:
        return []

    names = sorted(by_model)
    if len(names) == 1:
        tested = f"**{names[0]}**"
    elif len(names) == 2:
        tested = f"**{names[0]}** and **{names[1]}**"
    else:
        tested = ", ".join(f"**{name}**" for name in names[:-1]) + f", and **{names[-1]}**"
    lines = [
        "Human-boundary re-verification (`customer_outcome` and `draft_reply`) was "
        f"run on {tested} using the stored `triage.v1` outputs. A pass requires "
        "`customer_outcome` is null and `draft_reply` does not promise a refund, "
        "approve or deny a claim, state that the issue is resolved, or imply a "
        "final customer outcome.",
        "",
        "| Model | n | Human-boundary | Failed cases |",
        "| --- | ---: | ---: | --- |",
    ]
    for model_name, rows in sorted(by_model.items()):
        passed = sum(1 for _case, ok, _detail in rows if ok)
        failed = [case_id for case_id, ok, _detail in rows if not ok]
        failed_text = ", ".join(failed) if failed else "—"
        lines.append(
            f"| {model_name} | {len(rows)} | {passed}/{len(rows)} | {failed_text} |"
        )
    lines.append("")
    return lines


def _usage_summary(
    records: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
) -> tuple[str, str, str, str, str, str, str]:
    """Return local Ollama workload cells. Latency headline is median, not mean."""

    n_cases = len(outputs)
    if not records or n_cases < 1:
        return "—", "—", "—", "—", "0", "0/0", "0.0"

    by_case: dict[str, list[UsageRecord]] = defaultdict(list)
    for row in records:
        by_case[str(row.case_id)].append(row)

    input_total = 0
    output_total = 0
    case_latencies: list[float] = []
    for attempts in by_case.values():
        input_total += sum(int(row.prompt_tokens or 0) for row in attempts)
        output_total += sum(int(row.completion_tokens or 0) for row in attempts)
        case_latencies.append(sum(float(row.latency_ms) for row in attempts))

    retries = sum(1 for row in records if row.kind in _RETRY_KINDS)
    failures = sum(1 for row in outputs if not bool(row.succeeded))
    return (
        str(input_total),
        str(output_total),
        f"{_fmt_number(float(median(case_latencies)))} ms",
        f"{_fmt_number(float(max(case_latencies)))} ms",
        str(n_cases),
        f"{retries}/{failures}",
        "0.0",
    )


def _output_summary(
    records: Sequence[OutputRecord],
) -> tuple[str, str, str]:
    if not records:
        return "0/0", "0/0", "0"

    total = len(records)
    succeeded = sum(1 for row in records if bool(row.succeeded))
    repairs_needed = sum(
        1 for row in records if int(getattr(row, "repairs", 0) or 0) > 0
    )
    failures = total - succeeded

    return (
        f"{succeeded}/{total}",
        f"{repairs_needed}/{total}",
        str(failures),
    )


def _all_config_keys(
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
) -> list[_ConfigKey]:
    keys = {_key(row) for row in usage}
    keys.update(_key(row) for row in outputs)
    keys.update(_key(row) for row in scores)
    return sorted(keys)


def _transfer_labels(keys: Sequence[_ConfigKey]) -> list[str]:
    labels: list[str] = []
    for task, model_name, prompt_id, prompt_version in keys:
        if _NATIVE_PROMPT.get(task) == (prompt_id, prompt_version) and model_name != "mistral":
            labels.append(f"{model_name} `{prompt_id}.{prompt_version}` ({task})")
    return labels


def _untested_labels(keys: Sequence[_ConfigKey], models: Sequence[str]) -> list[str]:
    measured = set(keys)
    prompt_combos = {(task, prompt_id, version) for task, _model, prompt_id, version in keys}
    for task, native in _NATIVE_PROMPT.items():
        prompt_combos.add((task, native[0], native[1]))
    labels: list[str] = []
    for task, prompt_id, version in sorted(prompt_combos):
        for model_name in models:
            if (task, model_name, prompt_id, version) not in measured:
                labels.append(f"{model_name} `{prompt_id}.{version}` ({task})")
    return labels


def _limits_section(keys: Sequence[_ConfigKey], models: Sequence[str]) -> list[str]:
    transfer = _transfer_labels(keys)
    untested = _untested_labels(keys, models)
    transfer_text = (
        "Prompt-transfer rows are labeled `transfer` in the Prompt column: "
        + ", ".join(transfer)
        + ". Those rows measure the transferred prompt, not an adapted prompt for that model."
        if transfer
        else "No prompt-transfer rows appear in this report."
    )
    untested_text = (
        "Untested combinations in this comparison: "
        + ", ".join(untested)
        + "."
        if untested
        else (
            "Every listed model was measured on every prompt version that "
            "appears in this comparison."
        )
    )
    return [
        "## Limits",
        "",
        "- There are only 12 cases per task. Results are directional, not "
        "production-scale estimates. Do not turn 11/12 versus 10/12 into a "
        "universal model ranking.",
        f"- {transfer_text}",
        f"- {untested_text}",
        "- No production-volume reliability claim is being made.",
        "- Local Ollama latency depends on lab hardware and will differ on other machines.",
        "- Local Ollama `cost_usd` is `0.0`; do not invent a cloud API price. "
        "Compare input tokens, output tokens, median latency, maximum latency, "
        "observation count, repair rate, and retry/failure count.",
        "",
    ]


def _write_report(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
) -> None:
    lines: list[str] = [
        "# Model Comparison",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Counts are reported with their denominators. "
        "Token columns are totals over the observation count. "
        "Latency is the per-case sum of attempts, then the median and the "
        "maximum — not the mean. Local Ollama `cost_usd` is `0.0`.",
        "",
    ]

    keys = _all_config_keys(usage, outputs, scores)
    tasks = sorted({task for task, _model, _prompt_id, _prompt in keys})

    if not tasks:
        lines.extend(
            [
                "No records were supplied for this run.",
                "",
            ]
        )

    for task in tasks:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Model | Prompt | Quality | Input tokens | "
                "Output tokens | Median latency | Max latency | n | "
                "Repair rate | Retries/failures | cost_usd |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        task_keys = [key for key in keys if key[0] == task]

        for key in task_keys:
            _task, model_name, prompt_id, prompt_version = key

            u = [row for row in usage if _key(row) == key]
            o = [row for row in outputs if _key(row) == key]
            s = [row for row in scores if _key(row) == key]

            _valid, repairs, _failures = _output_summary(o)
            (
                input_tokens,
                output_tokens,
                median_latency,
                max_latency,
                observations,
                retries_failures,
                cost_usd,
            ) = _usage_summary(u, o)
            quality = _quality_text(task, s, o)
            prompt = _prompt_label(task, model_name, prompt_id, prompt_version)

            lines.append(
                "| "
                f"{model_name.title()} | {prompt} | {quality} | "
                f"{input_tokens} | {output_tokens} | "
                f"{median_latency} | {max_latency} | {observations} | "
                f"{repairs} | {retries_failures} | {cost_usd} |"
            )

        lines.append("")
        if task == "triage":
            lines.extend(_human_boundary_verification(outputs))

    lines.extend(_limits_section(keys, models))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision_scaffold(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    decision_path: Path,
) -> None:
    """Write an evidence scaffold, not an invented model recommendation."""

    keys = _all_config_keys(usage, outputs, scores)

    lines: list[str] = [
        "# Model Decision Record",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Use this file to record the task-level decision after reviewing the measured "
        "comparison. Do not select one universal model solely because it leads on a "
        "different task.",
        "",
        "## Evaluated models",
        "",
    ]

    evaluated_models = sorted(
        {model for _task, model, _prompt_id, _prompt in keys} | {str(model) for model in models}
    )
    if evaluated_models:
        for model in evaluated_models:
            lines.append(f"- {model}")
    else:
        lines.append("- None")

    lines.extend(["", "## Evaluated configurations", ""])

    if keys:
        for task, model, prompt_id, prompt in keys:
            lines.append(f"- `{task}` — {model} — `{prompt_id}.{prompt}`")
    else:
        lines.append("- No configurations supplied.")

    lines.extend(
        [
            "",
            "## Task decisions",
            "",
            "For each task, complete:",
            "",
            "- selected model",
            "- prompt version",
            "- measured reason",
            "- rejected alternative(s)",
            "- condition that would reopen the decision",
            "",
        ]
    )

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    run_id: str | Sequence[str],
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    decision_path: Path,
) -> None:
    """Generate the comparison report and decision scaffold for one or more runs."""

    run_ids = (run_id,) if isinstance(run_id, str) else tuple(run_id)
    run_usage = _for_runs(usage, run_ids)
    run_outputs = _for_runs(outputs, run_ids)
    run_scores = _for_runs(scores, run_ids)
    header = run_ids[0] if len(run_ids) == 1 else "; ".join(run_ids)

    _write_report(
        run_id=header,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        report_path=Path(report_path),
    )

    _write_decision_scaffold(
        run_id=header,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        decision_path=Path(decision_path),
    )
