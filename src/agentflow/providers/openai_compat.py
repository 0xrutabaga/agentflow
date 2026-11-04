"""Optional provider for any OpenAI-compatible ``/chat/completions`` endpoint.

Never used by the tests or the demo against a real network: the tests inject an
``httpx.MockTransport``. Configure it with environment variables:

``AGENTFLOW_OPENAI_BASE_URL``  for example ``https://api.openai.com/v1`` or a local server
``AGENTFLOW_OPENAI_API_KEY``   bearer token (optional for local servers)
``AGENTFLOW_OPENAI_MODEL``     model name
``AGENTFLOW_OPENAI_TIMEOUT_S`` request timeout in seconds (default 60)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from agentflow.messages import Message, ModelResponse, ToolCallRequest
from agentflow.providers.base import ProviderError, TransientProviderError

TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _parse_arguments(raw: Any) -> Any:
    """Decode the JSON argument string. On failure return it unchanged so that schema
    validation rejects it and the model is asked to repair it."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> OpenAICompatibleProvider:
        env = os.environ if env is None else env
        base_url = env.get("AGENTFLOW_OPENAI_BASE_URL")
        model = env.get("AGENTFLOW_OPENAI_MODEL")
        if not base_url or not model:
            raise ProviderError(
                "set AGENTFLOW_OPENAI_BASE_URL and AGENTFLOW_OPENAI_MODEL to use the HTTP provider"
            )
        return cls(
            base_url=base_url,
            model=model,
            api_key=env.get("AGENTFLOW_OPENAI_API_KEY") or None,
            timeout_s=float(env.get("AGENTFLOW_OPENAI_TIMEOUT_S", "60")),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        output_schema: dict[str, Any],
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
        }
        if tools:
            body["tools"] = [{"type": "function", "function": dict(spec)} for spec in tools]
        try:
            response = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"model request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise TransientProviderError(f"model request failed: {exc}") from exc
        if response.status_code in TRANSIENT_STATUS:
            raise TransientProviderError(f"model endpoint returned HTTP {response.status_code}")
        if response.status_code >= 400:
            # The body is not echoed: some servers include the request (and so the key) in it.
            raise ProviderError(f"model endpoint returned HTTP {response.status_code}")
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from model endpoint: {exc!r}") from exc
        calls = tuple(
            ToolCallRequest(
                id=str(raw.get("id") or f"call_{index}"),
                name=str(raw.get("function", {}).get("name", "")),
                arguments=_parse_arguments(raw.get("function", {}).get("arguments")),
            )
            for index, raw in enumerate(message.get("tool_calls") or [])
        )
        usage = {k: int(v) for k, v in (payload.get("usage") or {}).items() if isinstance(v, int)}
        return ModelResponse(content=message.get("content"), tool_calls=calls, usage=usage)
