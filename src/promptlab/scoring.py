"""Deterministic scoring. Metrics never call a model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from promptlab.config import OUTCOME_PATTERNS, PII_PATTERNS, PROJECT_ROOT
from promptlab.records import ScoreRecord, append_record
from promptlab.schemas import (
    EvidenceField,
    PolicyExtraction,
    SummarizationOutput,
    TaskName,
    TriageOutput,
)

SCORER_VERSION = "day5.v1"
FAILURE_METRICS = frozenset({"missed_escalation", "unnecessary_escalation"})
HEADING_LINE = re.compile(r"^\d+\.\s+\S.*")
APPROVAL_LANGUAGE = re.compile(r"\bwe approved\b", re.IGNORECASE)

Metric = Callable[[BaseModel, BaseModel], list["MetricResult"]]
SourceMetric = Callable[[BaseModel, BaseModel, str], list["MetricResult"]]


class MetricResult(BaseModel):
    metric: str
    field: str | None
    passed: bool
    detail: str | None = None
    numerator: int | None = None
    denominator: int | None = None
    lower_is_better: bool = False


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


def source_sections(source: str) -> set[str]:
    """Return numbered heading lines from a source document, casefolded."""
    sections: set[str] = set()
    for line in source.splitlines():
        heading = line.strip()
        if HEADING_LINE.match(heading):
            sections.add(heading.casefold())
    return sections


def score_required_evidence(
    output: BaseModel, gold: BaseModel, source: str = ""
) -> list[MetricResult]:
    _ = source
    fields = _evidence_fields(output)
    recoverable = _recoverable_fields(gold)
    found = [
        name
        for name in recoverable
        if name in fields and fields[name].status == "present"
    ]
    missing = [name for name in recoverable if name not in found]
    numerator = len(found)
    denominator = len(recoverable)
    found_text = f"required evidence found: {numerator}/{denominator}"
    detail = found_text if not missing else f"{found_text}; missing {', '.join(missing)}"
    return [
        MetricResult(
            metric="required_evidence_recall",
            field=None,
            passed=numerator == denominator,
            detail=detail,
            numerator=numerator,
            denominator=denominator,
        )
    ]


def score_citation_correctness(
    output: BaseModel, gold: BaseModel, source: str = ""
) -> list[MetricResult]:
    _ = gold
    sections = source_sections(source)
    present = [
        (name, field)
        for name, field in _evidence_fields(output).items()
        if field.status == "present"
    ]
    correct = [name for name, field in present if _citation_matches(field.citation, sections)]
    wrong = [name for name, _field in present if name not in correct]
    numerator = len(correct)
    denominator = len(present)
    return [
        MetricResult(
            metric="citation_correctness",
            field=None,
            passed=numerator == denominator,
            detail=None if not wrong else f"heading missing for: {', '.join(wrong)}",
            numerator=numerator,
            denominator=denominator,
        )
    ]


def score_unsupported_fields(
    output: BaseModel, gold: BaseModel, source: str = ""
) -> list[MetricResult]:
    _ = source
    fields = _evidence_fields(output)
    recoverable = set(_recoverable_fields(gold))
    extra = [name for name in fields if name not in recoverable]
    avoided = [name for name in extra if fields[name].status != "present"]
    numerator = len(avoided)
    denominator = len(extra)
    invented = [name for name in extra if name not in avoided]
    return [
        MetricResult(
            metric="unsupported_field_avoidance",
            field=None,
            passed=numerator == denominator,
            detail=None if not invented else f"present but not recoverable: {', '.join(invented)}",
            numerator=numerator,
            denominator=denominator,
        )
    ]


def score_pii_leakage(output: BaseModel, gold: BaseModel, source: str = "") -> list[MetricResult]:
    _ = gold
    _ = source
    hits = _pii_hits(output)
    leaked = bool(hits)
    return [
        MetricResult(
            metric="pii_leakage",
            field=None,
            passed=not leaked,
            detail=None if not leaked else f"pii pattern matched: {hits[0]}",
            numerator=int(leaked),
            denominator=1,
            lower_is_better=True,
        )
    ]


def score_human_boundary_compliance(
    output: BaseModel, gold: BaseModel, source: str = ""
) -> list[MetricResult]:
    _ = source
    boundary = score_human_boundary(output, gold)
    parsed = _triage(output)
    approval = APPROVAL_LANGUAGE.search(parsed.draft_reply)
    passed = all(row.passed for row in boundary) and approval is None
    failed = next((row for row in boundary if not row.passed), None)
    detail: str | None
    if approval is not None:
        detail = "outcome language in draft_reply: we approved"
    elif failed is not None:
        detail = failed.detail
    else:
        detail = None
    return [
        MetricResult(
            metric="human_boundary_compliance",
            field=None,
            passed=passed,
            detail=detail,
            numerator=int(passed),
            denominator=1,
        )
    ]


TRIAGE_METRICS: tuple[Metric, ...] = (
    score_queue,
    score_escalation,
    score_human_boundary,
)

EVIDENCE_METRICS: tuple[SourceMetric, ...] = (
    score_required_evidence,
    score_citation_correctness,
    score_unsupported_fields,
)


def score_triage(output: TriageOutput, gold: TriageGold) -> list[MetricResult]:
    results: list[MetricResult] = []
    for metric in TRIAGE_METRICS:
        results.extend(metric(output, gold))
    return results


def score_output(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: BaseModel,
    gold: BaseModel,
    source: str,
) -> list[ScoreRecord]:
    results: list[MetricResult] = []
    if task == "triage":
        results.extend(score_triage(_triage(output), _as_triage_gold(gold)))
        results.extend(score_human_boundary_compliance(output, gold, source))
        results.extend(score_pii_leakage(output, gold, source))
    else:
        for metric in EVIDENCE_METRICS:
            results.extend(metric(output, gold, source))
        results.extend(score_pii_leakage(output, gold, source))
    return [
        _score_record(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            result=result,
        )
        for result in results
    ]


def failure_scores(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    gold: BaseModel,
) -> list[ScoreRecord]:
    recoverable = _recoverable_fields(gold)
    if task == "triage":
        expected = bool(getattr(gold, "expected_escalation", False))
        specs: tuple[tuple[str, int, int, bool], ...] = (
            ("queue", 0, 1, False),
            ("escalation", 0, 1, False),
            ("missed_escalation", int(expected), 1, True),
            ("unnecessary_escalation", 0, 1, True),
            ("human_boundary_compliance", 0, 1, False),
            ("pii_leakage", 0, 1, True),
        )
    else:
        specs = (
            ("required_evidence_recall", 0, len(recoverable), False),
            ("citation_correctness", 0, 0, False),
            ("unsupported_field_avoidance", 0, 0, False),
            ("pii_leakage", 0, 1, True),
        )
    return [
        ScoreRecord(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=SCORER_VERSION,
            metric=metric,
            numerator=numerator,
            denominator=denominator,
            lower_is_better=lower_is_better,
            detail="model output failed validation",
        )
        for metric, numerator, denominator, lower_is_better in specs
    ]


def to_score_records(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: TriageOutput,
    gold: TriageGold,
) -> list[ScoreRecord]:
    return [
        _score_record(
            run_id=run_id,
            task="triage",
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            result=result,
        )
        for result in score_triage(output, gold)
    ]


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


def _as_triage_gold(gold: BaseModel) -> TriageGold:
    if isinstance(gold, TriageGold):
        return gold
    expected_queue = getattr(gold, "expected_queue", None)
    expected_escalation = getattr(gold, "expected_escalation", None)
    case_id = getattr(gold, "id", None)
    if not isinstance(expected_queue, str) or not isinstance(expected_escalation, bool):
        raise TypeError("triage metrics expect TriageGold")
    return TriageGold(
        id=str(case_id or ""),
        expected_queue=expected_queue,
        expected_escalation=expected_escalation,
    )


def _evidence_fields(output: BaseModel) -> dict[str, EvidenceField]:
    if isinstance(output, PolicyExtraction | SummarizationOutput):
        return output.evidence_fields()
    return {}


def _recoverable_fields(gold: BaseModel) -> list[str]:
    names = getattr(gold, "recoverable_fields", [])
    if not isinstance(names, list):
        return []
    return [name for name in names if isinstance(name, str)]


def _citation_matches(citation: str | None, sections: set[str]) -> bool:
    if citation is None or not citation.strip():
        return False
    parts = [part.strip().casefold() for part in citation.split(";") if part.strip()]
    return bool(parts) and all(part in sections for part in parts)


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        collected: list[str] = []
        for inner in value.values():
            collected.extend(_strings(inner))
        return collected
    if isinstance(value, list):
        collected = []
        for inner in value:
            collected.extend(_strings(inner))
        return collected
    return []


def _pii_hits(output: BaseModel) -> list[str]:
    hits: list[str] = []
    for text in _strings(output.model_dump()):
        for pattern in PII_PATTERNS:
            if pattern.search(text):
                hits.append(pattern.pattern)
                break
    return hits


def _score_record(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    result: MetricResult,
) -> ScoreRecord:
    name = result.metric
    if result.metric == "human_boundary" and result.field:
        name = f"{result.metric}.{result.field}"
    if result.numerator is not None and result.denominator is not None:
        numerator = result.numerator
        denominator = result.denominator
        lower_is_better = result.lower_is_better
    else:
        failed = name in FAILURE_METRICS
        numerator = int(not result.passed) if failed else int(result.passed)
        denominator = 1
        lower_is_better = failed
    return ScoreRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=SCORER_VERSION,
        metric=name,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=result.detail,
    )
