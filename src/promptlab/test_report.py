from decimal import Decimal
from pathlib import Path
from typing import Literal

from promptlab.config import Settings
from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.report import write_reports

Kind = Literal["primary", "transport_retry", "repair", "repair_retry"]


def test_report_is_generated_from_records(tmp_path: object) -> None:
    model = Settings.from_env().models["mistral"]
    root = Path(str(tmp_path))
    usage = [
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name=model.logical_name,
            model_id=model.model_id,
            prompt_id="triage",
            prompt_version="triage-mistral-v1",
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=100,
            completion_tokens=25,
            latency_ms=125.0,
            cost_usd=Decimal("0"),
        )
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name=model.logical_name,
            model_id=model.model_id,
            prompt_id="triage",
            prompt_version="triage-mistral-v1",
            succeeded=True,
            repairs=0,
            output={"queue": "card_dispute"},
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name=model.logical_name,
            model_id=model.model_id,
            prompt_id="triage",
            prompt_version="triage-mistral-v1",
            scorer_version="2.0.0",
            metric="queue",
            numerator=1,
            denominator=1,
        )
    ]
    report = root / "comparison.md"
    decision = root / "model-decision.md"
    write_reports(
        run_id="demo",
        models=[model.logical_name],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "1/1" in text
    assert "triage.triage-mistral-v1" in text
    assert "Input tokens" in text
    assert "Output tokens" in text
    assert "Median latency" in text
    assert "Max latency" in text
    assert "| n |" in text
    assert "Repair rate" in text
    assert "Retries/failures" in text
    assert "cost_usd" in text
    assert "0.0" in text
    assert "routing:" in text
    assert "Mean latency" not in text
    assert "There are only 12 cases per task" in text
    assert "directional, not production-scale estimates" in text
    assert "universal model ranking" in text
    assert "No production-volume reliability claim" in text
    assert "Local Ollama latency depends on lab hardware" in text
    assert "mistral" in decision.read_text(encoding="utf-8")


def _usage(
    *,
    case_id: str,
    latency_ms: float,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    kind: Kind = "primary",
    attempt: int = 1,
) -> UsageRecord:
    model = Settings.from_env().models["mistral"]
    return UsageRecord(
        run_id="demo",
        task="triage",
        case_id=case_id,
        model_name=model.logical_name,
        model_id=model.model_id,
        prompt_id="triage",
        prompt_version="v1",
        attempt=attempt,
        kind=kind,
        status="success",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=Decimal("0"),
    )


def _output(*, case_id: str, succeeded: bool = True, repairs: int = 0) -> OutputRecord:
    model = Settings.from_env().models["mistral"]
    return OutputRecord(
        run_id="demo",
        task="triage",
        case_id=case_id,
        model_name=model.logical_name,
        model_id=model.model_id,
        prompt_id="triage",
        prompt_version="v1",
        succeeded=succeeded,
        repairs=repairs,
        output={"queue": "card_dispute"} if succeeded else None,
    )


def test_report_uses_median_latency_not_mean(tmp_path: object) -> None:
    model = Settings.from_env().models["mistral"]
    root = Path(str(tmp_path))
    usage = [
        _usage(case_id="T01", latency_ms=10.0),
        _usage(case_id="T02", latency_ms=20.0),
        _usage(case_id="T03", latency_ms=1000.0),
    ]
    outputs = [
        _output(case_id="T01"),
        _output(case_id="T02"),
        _output(case_id="T03"),
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name=model.logical_name,
            model_id=model.model_id,
            prompt_id="triage",
            prompt_version="v1",
            scorer_version="2.0.0",
            metric="queue",
            numerator=1,
            denominator=1,
        )
    ]
    report = root / "comparison.md"
    write_reports(
        run_id="demo",
        models=[model.logical_name],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=root / "model-decision.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "20 ms" in text
    assert "1000 ms" in text
    assert "343" not in text
    assert "| 3 |" in text
    assert "| 0/3 |" in text
    assert "| 0/0 |" in text
    assert "| 0.0 |" in text
    assert "30" in text
    assert "15" in text


def test_limits_identify_transfer_and_untested_rows(tmp_path: object) -> None:
    settings = Settings.from_env()
    mistral = settings.models["mistral"]
    qwen = settings.models["qwen"]
    root = Path(str(tmp_path))

    def usage(model_name: str, model_id: str, version: str, case_id: str) -> UsageRecord:
        return UsageRecord(
            run_id="demo",
            task="extraction",
            case_id=case_id,
            model_name=model_name,
            model_id=model_id,
            prompt_id="extract",
            prompt_version=version,
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=10.0,
            cost_usd=Decimal("0"),
        )

    def output(model_name: str, model_id: str, version: str, case_id: str) -> OutputRecord:
        return OutputRecord(
            run_id="demo",
            task="extraction",
            case_id=case_id,
            model_name=model_name,
            model_id=model_id,
            prompt_id="extract",
            prompt_version=version,
            succeeded=True,
            repairs=0,
            output={"document_status": "valid"},
        )

    write_reports(
        run_id="demo",
        models=["mistral", "qwen"],
        usage=[
            usage(mistral.logical_name, mistral.model_id, "v2", "E01"),
            usage(qwen.logical_name, qwen.model_id, "v2", "E01"),
            usage(qwen.logical_name, qwen.model_id, "v3", "E01"),
        ],
        outputs=[
            output(mistral.logical_name, mistral.model_id, "v2", "E01"),
            output(qwen.logical_name, qwen.model_id, "v2", "E01"),
            output(qwen.logical_name, qwen.model_id, "v3", "E01"),
        ],
        scores=[],
        report_path=root / "comparison.md",
        decision_path=root / "model-decision.md",
    )
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "qwen `extract.v2` (extraction)" in text
    assert "mistral `extract.v3` (extraction)" in text
    assert "labeled `transfer`" in text
    assert "Untested combinations in this comparison:" in text


def _triage_output_dict(draft_reply: str) -> dict[str, object]:
    return {
        "queue": "card_dispute",
        "escalation_required": False,
        "confidence": 0.8,
        "rationale": "Recognized duplicate charge.",
        "draft_reply": draft_reply,
        "human_review_required": True,
        "customer_outcome": None,
    }


def test_report_rechecks_human_boundary_on_mistral_and_qwen(
    tmp_path: object,
) -> None:
    settings = Settings.from_env()
    root = Path(str(tmp_path))

    def row(
        model_name: str, model_id: str, case_id: str, draft: str
    ) -> tuple[UsageRecord, OutputRecord]:
        usage = UsageRecord(
            run_id="demo",
            task="triage",
            case_id=case_id,
            model_name=model_name,
            model_id=model_id,
            prompt_id="triage",
            prompt_version="v1",
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=10.0,
            cost_usd=Decimal("0"),
        )
        output = OutputRecord(
            run_id="demo",
            task="triage",
            case_id=case_id,
            model_name=model_name,
            model_id=model_id,
            prompt_id="triage",
            prompt_version="v1",
            succeeded=True,
            repairs=0,
            output=_triage_output_dict(draft),
        )
        return usage, output

    mistral = settings.models["mistral"]
    qwen = settings.models["qwen"]
    u1, o1 = row(mistral.logical_name, mistral.model_id, "T01", "We will review the charge.")
    u2, o2 = row(qwen.logical_name, qwen.model_id, "T01", "We will refund you today.")
    write_reports(
        run_id="demo",
        models=["mistral", "qwen"],
        usage=[u1, u2],
        outputs=[o1, o2],
        scores=[],
        report_path=root / "comparison.md",
        decision_path=root / "model-decision.md",
    )
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "Human-boundary re-verification" in text
    assert "**mistral**" in text
    assert "**qwen**" in text
    assert "| mistral | 1 | 1/1 | — |" in text
    assert "| qwen | 1 | 0/1 | T01 |" in text


