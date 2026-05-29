#!/usr/bin/env python3
"""Single OpenAI-compatible client layer for all GLM calls."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jsonschema import ValidationError, validate
from openai import OpenAI


@dataclass
class GLMCallRecord:
    prompt_hash: str
    model: str
    tokens: int | None
    latency_seconds: float
    retry_count: int


class GLMClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("GLM_MODEL", "glm-4.5")
        self.base_url = base_url or os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.api_key = api_key or os.environ["ZHIPU_API_KEY"]
        timeout_seconds = float(os.environ.get("GLM_TIMEOUT_SECONDS", "180"))
        self.request_timeout_seconds = timeout_seconds
        self.network_retry_count = int(os.environ.get("GLM_NETWORK_RETRIES", "2"))
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 30.0)),
            ),
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        thinking: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> tuple[dict[str, Any], GLMCallRecord]:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "resistagent_response", "schema": json_schema},
            },
        }
        extra_body = None
        if thinking is not None:
            extra_body = {"thinking": thinking}
        prompt_envelope = {"payload": payload, "extra_body": extra_body}
        prompt_hash = hashlib.sha256(json.dumps(prompt_envelope, sort_keys=True).encode("utf-8")).hexdigest()
        total_latency = 0.0
        total_tokens = 0
        retry_count = 0
        response = self._create_completion(model=self.model, payload=payload, extra_body=extra_body)
        total_latency += response["latency"]
        total_tokens += response["tokens"]
        content = response["content"]
        try:
            parsed = self._parse_and_validate_json(content, json_schema)
        except (json.JSONDecodeError, ValidationError, ValueError):
            parsed = None
        while parsed is None and retry_count < max_retries:
            retry_count += 1
            repair_payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON formatter. "
                            "Convert the assistant response into valid JSON that matches the provided schema exactly. "
                            "Return JSON only with no markdown fences or commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "schema": json_schema,
                                "invalid_response": content,
                            },
                            sort_keys=True,
                            ensure_ascii=True,
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "resistagent_response", "schema": json_schema},
                },
            }
            repair_response = self._create_completion(model=self.model, payload=repair_payload, extra_body={"thinking": {"type": "disabled"}})
            total_latency += repair_response["latency"]
            total_tokens += repair_response["tokens"]
            content = repair_response["content"]
            try:
                parsed = self._parse_and_validate_json(content, json_schema)
            except (json.JSONDecodeError, ValidationError, ValueError):
                parsed = None
        if parsed is None:
            excerpt = (content or "").strip().replace("\n", " ")[:400]
            raise ValueError(f"GLMClient could not obtain valid schema-conformant JSON. Last content excerpt: {excerpt}")
        record = GLMCallRecord(
            prompt_hash=prompt_hash,
            model=self.model,
            tokens=total_tokens or None,
            latency_seconds=total_latency,
            retry_count=retry_count,
        )
        return parsed, record

    def _create_completion(self, *, model: str, payload: dict[str, Any], extra_body: dict[str, Any] | None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.network_retry_count + 1):
            start = time.time()
            try:
                response = self.client.chat.completions.create(model=model, extra_body=extra_body, **payload)
            except Exception as exc:
                last_error = exc
                if attempt >= self.network_retry_count:
                    raise
                time.sleep(min(8.0, 2.0 * (attempt + 1)))
                continue
            latency = time.time() - start
            usage = getattr(response, "usage", None)
            return {
                "content": response.choices[0].message.content or "",
                "tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                "latency": latency,
            }
        assert last_error is not None
        raise last_error

    def _parse_and_validate_json(self, content: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._extract_json_text(content)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Expected top-level JSON object.")
        validate(instance=parsed, schema=json_schema)
        return parsed

    def _extract_json_text(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            raise ValueError("Empty model content.")
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text
