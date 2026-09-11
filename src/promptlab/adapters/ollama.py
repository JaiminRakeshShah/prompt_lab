from __future__ import annotations

import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord, compute_cost

MAX_ATTEMPTS = 3


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._settings = Settings.from_env()

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        if not any(config.model_id == self.model_id for config in self._settings.models.values()):
            record = self._record(
                request=request,
                run_id=run_id,
                attempt=1,
                latency_ms=0,
                payload=None,
                error_type=UnknownModelError.__name__,
            )
            return CompletionResult(
                succeeded=False,
                text=None,
                error_type=UnknownModelError.__name__,
                records=[record],
            )

        records: list[CallRecord] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    f"{self._settings.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_id,
                        "prompt": f"{request.system}\n\n{request.user_content}",
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_output_tokens,
                        },
                    },
                    timeout=180.0,
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        payload=None,
                        error_type=TransientProviderError.__name__,
                    )
                )
                if attempt < MAX_ATTEMPTS:
                    _backoff(attempt)
                    continue
                return CompletionResult(
                    succeeded=False,
                    text=None,
                    error_type=TransientProviderError.__name__,
                    records=records,
                )

            latency_ms = int((time.perf_counter() - started) * 1000)
            payload = response.json()
            text = _response_text(payload)

            if _is_transient_status(response.status_code):
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        payload=payload,
                        error_type=TransientProviderError.__name__,
                    )
                )
                if attempt < MAX_ATTEMPTS:
                    _backoff(attempt)
                    continue
                return CompletionResult(
                    succeeded=False,
                    text=text,
                    error_type=TransientProviderError.__name__,
                    records=records,
                )

            if response.status_code >= 400:
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        payload=payload,
                        error_type=PermanentProviderError.__name__,
                    )
                )
                return CompletionResult(
                    succeeded=False,
                    text=text,
                    error_type=PermanentProviderError.__name__,
                    records=records,
                )

            if payload.get("done_reason") == "length":
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        payload=payload,
                        error_type=TruncatedResponseError.__name__,
                    )
                )
                return CompletionResult(
                    succeeded=False,
                    text=text,
                    error_type=TruncatedResponseError.__name__,
                    records=records,
                )

            records.append(
                self._record(
                    request=request,
                    run_id=run_id,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    payload=payload,
                    error_type=None,
                )
            )
            return CompletionResult(
                succeeded=True,
                text=text,
                error_type=None,
                records=records,
            )

        raise RuntimeError("unreachable")

    def _record(
        self,
        *,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
        latency_ms: int,
        payload: dict[str, Any] | None,
        error_type: str | None,
    ) -> CallRecord:
        input_tokens = _int_field(payload, "prompt_eval_count")
        output_tokens = _int_field(payload, "eval_count")
        stop_reason = None
        text = None
        if payload is not None:
            done_reason = payload.get("done_reason")
            stop_reason = None if done_reason is None else str(done_reason)
            text = _response_text(payload)
        cost_usd = 0.0
        if error_type != UnknownModelError.__name__:
            cost_usd = compute_cost(self.model_id, input_tokens, output_tokens)
        return CallRecord(
            record_id=str(uuid.uuid4()),
            run_id=run_id,
            timestamp=datetime.now(UTC),
            provider="ollama",
            model_id=self.model_id,
            task=request.task,
            case_id=request.case_id,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            attempt=attempt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            stop_reason=stop_reason,
            error_type=error_type,
            response_text=text,
        )


def _is_transient_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _backoff(attempt: int) -> None:
    time.sleep((2 ** (attempt - 1)) + random.random())


def _response_text(payload: dict[str, Any]) -> str | None:
    response = payload.get("response")
    if isinstance(response, str):
        return response
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def _int_field(payload: dict[str, Any] | None, key: str) -> int:
    if payload is None:
        return 0
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0
