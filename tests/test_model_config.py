from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings


def _model_id(name: str) -> str:
    return Settings.from_env().models[name].model_id


def _request() -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id="case_001",
        prompt_id="baseline",
        prompt_version="v0",
        system="You are a helpful assistant.",
        user_content="Summarize this procedure.",
        temperature=0.0,
        max_output_tokens=64,
    )


class _Response:
    status_code = 200

    def json(self) -> dict[str, object]:
        return {
            "response": "ok",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "done_reason": "stop",
        }


def test_qwen_thinking_is_disabled_in_config() -> None:
    models = Settings.from_env().models
    assert models["qwen"].think is False
    assert models["mistral"].think is None


def _generate_body(monkeypatch: pytest.MonkeyPatch, model_name: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_post(*args: object, **kwargs: object) -> _Response:
        captured["json"] = kwargs["json"]
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    OllamaAdapter(model_id=_model_id(model_name)).complete(_request(), "think-config")
    body = captured["json"]
    assert isinstance(body, dict)
    return body


def test_qwen_generate_sends_think_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    body = _generate_body(monkeypatch, "qwen")
    assert body["think"] is False


def test_mistral_generate_omits_think(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    body = _generate_body(monkeypatch, "mistral")
    assert "think" not in body
