"""Deterministic Day 4 triage scoring. Metrics never call a model."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from promptlab.config import OUTCOME_PATTERNS, PROJECT_ROOT
from promptlab.records import ScoreRecord, append_record
from promptlab.schemas import TriageOutput

SCORER_VERSION = "day4.v1"
FAILURE_METRICS = frozenset({"missed_escalation", "unnecessary_escalation"})

Metric = Callable[[BaseModel, BaseModel], list["MetricResult"]]


class MetricResult(BaseModel):
    metric: str
    field: str | None
    passed: bool
    detail: str | None = None


class TriageGold(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    expected_queue: str
    expected_escalation: bool


def load_triage_gold() -> dict[str, TriageGold]:
    path = PROJECT_ROOT / "cases" / "gold" / "triage.jsonl"
    gold: dict[str, TriageGold] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = TriageGold.model_validate(json.loads(line))
        gold[row.id] = row
    return gold


def score_queue(output: BaseModel, gold: BaseModel) -> list[MetricResult]:
    predicted = _triage(output).queue
    expected = _gold(gold).expected_queue
    passed = predicted == expected
    return [
        MetricResult(
            metric="queue",
            field="queue",
            passed=passed,
            detail=None if passed else f"predicted {predicted}, gold {expected}",
        )
    ]


def score_escalation(output: BaseModel, gold: BaseModel) -> list[MetricResult]:
    predicted = _triage(output).escalation_required
    expected = _gold(gold).expected_escalation
    missed = expected and not predicted
    unnecessary = predicted and not expected
    return [
        MetricResult(
            metric="escalation",
            field="escalation_required",
            passed=predicted == expected,
            detail=None if predicted == expected else f"predicted {predicted}, gold {expected}",
        ),
        MetricResult(
            metric="missed_escalation",
            field="escalation_required",
            passed=not missed,
            detail="gold required escalation; model did not" if missed else None,
        ),
        MetricResult(
            metric="unnecessary_escalation",
            field="escalation_required",
            passed=not unnecessary,
            detail="model required escalation; gold did not" if unnecessary else None,
        ),
    ]


def score_human_boundary(output: BaseModel, gold: BaseModel) -> list[MetricResult]:
    _ = gold
    parsed = _triage(output)
    outcome_ok = parsed.customer_outcome is None
    hits = [pattern.pattern for pattern in OUTCOME_PATTERNS if pattern.search(parsed.draft_reply)]
    return [
        MetricResult(
            metric="human_boundary",
            field="customer_outcome",
            passed=outcome_ok,
            detail=None if outcome_ok else "customer_outcome is not null",
        ),
        MetricResult(
            metric="human_boundary",
            field="draft_reply",
            passed=not hits,
            detail=None if not hits else f"outcome language in draft_reply: {hits[0]}",
        ),
    ]


TRIAGE_METRICS: tuple[Metric, ...] = (
    score_queue,
    score_escalation,
    score_human_boundary,
)


def score_triage(output: TriageOutput, gold: TriageGold) -> list[MetricResult]:
    results: list[MetricResult] = []
    for metric in TRIAGE_METRICS:
        results.extend(metric(output, gold))
    return results


def to_score_records(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: TriageOutput,
    gold: TriageGold,
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for result in score_triage(output, gold):
        name = result.metric
        if result.metric == "human_boundary" and result.field:
            name = f"{result.metric}.{result.field}"
        failed = name in FAILURE_METRICS
        numerator = int(not result.passed) if failed else int(result.passed)
        records.append(
            ScoreRecord(
                run_id=run_id,
                task="triage",
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                scorer_version=SCORER_VERSION,
                metric=name,
                numerator=numerator,
                denominator=1,
                lower_is_better=failed,
                detail=result.detail,
            )
        )
    return records


def write_scores(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: TriageOutput,
    gold: TriageGold,
) -> list[ScoreRecord]:
    records = to_score_records(
        run_id=run_id,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        output=output,
        gold=gold,
    )
    path = Path("runs") / f"{run_id}.scores.jsonl"
    for record in records:
        append_record(path, record)
    return records


def _triage(output: BaseModel) -> TriageOutput:
    if not isinstance(output, TriageOutput):
        raise TypeError("triage metrics expect TriageOutput")
    return output


def _gold(gold: BaseModel) -> TriageGold:
    if not isinstance(gold, TriageGold):
        raise TypeError("triage metrics expect TriageGold")
    return gold
