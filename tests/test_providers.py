"""Fake model and the OpenAI-compatible provider. No network: httpx.MockTransport only."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agentflow.messages import Message, ModelResponse, ToolCallRequest
from agentflow.providers.base import ProviderError, TransientProviderError
from agentflow.providers.fake import ScriptedModel
from agentflow.providers.openai_compat import OpenAICompatibleProvider

TOOLS = [{"name": "search", "description": "d", "parameters": {"type": "object"}}]
HISTORY = [
    Message(role="system", content="s"),
    Message(role="user", content="u"),
    Message(role="assistant", tool_calls=(ToolCallRequest("c1", "search", {"q": "x"}),)),
    Message(role="tool", content='{"ok": true}', tool_call_id="c1", name="search"),
]


def test_scripted_model_is_selected_by_conversation_not_call_count() -> None:
    model = ScriptedModel([ModelResponse(content="first"), ModelResponse(content="second")])
    assert model.complete(HISTORY[:2], [], {}).content == "first"
    assert model.complete(HISTORY, [], {}).content == "second"
    assert model.complete(HISTORY, [], {}).content == "second"  # same history, same answer
    with pytest.raises(ProviderError):
        model.complete([*HISTORY, Message(role="assistant", content="x")], [], {})


def _provider(handler: Any, api_key: str | None = "test-key") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://llm.invalid/v1",
        model="m",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
    )


def _reply(message: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, json={"choices": [{"message": message}], "usage": {"total_tokens": 7}}
    )


def test_request_shape_and_tool_call_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        function = {"name": "search", "arguments": '{"q": "db"}'}
        return _reply({"content": None, "tool_calls": [{"id": "x1", "function": function}]})

    response = _provider(handler).complete(HISTORY, TOOLS, {})
    assert seen["url"] == "https://llm.invalid/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "m"
    assert body["tools"] == [{"type": "function", "function": TOOLS[0]}]
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": '{"q": "x"}',
    }
    assert body["messages"][3] == {"role": "tool", "content": '{"ok": true}', "tool_call_id": "c1"}
    assert response.tool_calls == (ToolCallRequest("x1", "search", {"q": "db"}),)
    assert response.usage == {"total_tokens": 7}


def test_malformed_arguments_are_passed_through_for_validation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        function = {"name": "search", "arguments": "{q: db"}
        return _reply({"tool_calls": [{"id": "x", "function": function}]})

    [call] = _provider(handler).complete(HISTORY, TOOLS, {}).tool_calls
    assert call.arguments == "{q: db"


def test_final_content_and_no_auth_header_without_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return _reply({"content": '{"answer": "x"}'})

    response = _provider(handler, api_key=None).complete(HISTORY, [], {})
    assert response.content == '{"answer": "x"}'
    assert response.tool_calls == ()


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_statuses_are_transient(status: int) -> None:
    provider = _provider(lambda r: httpx.Response(status, json={}))
    with pytest.raises(TransientProviderError, match=str(status)):
        provider.complete(HISTORY, [], {})


def test_client_errors_are_permanent_and_do_not_echo_the_body() -> None:
    provider = _provider(lambda r: httpx.Response(401, text="bad key test-key"))
    with pytest.raises(ProviderError) as info:
        provider.complete(HISTORY, [], {})
    assert not isinstance(info.value, TransientProviderError)
    assert "test-key" not in str(info.value)


def test_network_errors_and_timeouts_are_transient() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    for handler in (refuse, slow):
        with pytest.raises(TransientProviderError):
            _provider(handler).complete(HISTORY, [], {})


def test_unexpected_response_shape() -> None:
    provider = _provider(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        provider.complete(HISTORY, [], {})


def test_from_env() -> None:
    with pytest.raises(ProviderError, match="AGENTFLOW_OPENAI_BASE_URL"):
        OpenAICompatibleProvider.from_env({})
    env = {"AGENTFLOW_OPENAI_BASE_URL": "http://localhost:8000/v1", "AGENTFLOW_OPENAI_MODEL": "l"}
    provider = OpenAICompatibleProvider.from_env(
        env, transport=httpx.MockTransport(lambda r: _reply({"content": "{}"}))
    )
    assert provider.model == "l"
    assert provider.complete(HISTORY, [], {}).content == "{}"


def test_http_provider_drives_a_full_run(tmp_path: Any) -> None:
    """The runtime works end to end with the HTTP provider, including argument repair."""
    from conftest import make_agent, make_tool

    from agentflow.runtime import Runtime
    from agentflow.state import RunStatus

    replies = iter(
        [
            {"tool_calls": [{"id": "a", "function": {"name": "t", "arguments": "not json"}}]},
            {"tool_calls": [{"id": "b", "function": {"name": "t", "arguments": "{}"}}]},
            {"content": '{"answer": "ok"}'},
        ]
    )
    provider = _provider(lambda r: _reply(next(replies)))
    agent = make_agent([], [make_tool("t", lambda: {"v": 1})], provider=provider)
    runtime = Runtime(tmp_path, sleep=lambda _: None)
    run_id = runtime.create("tests:inline", "goal", agent=agent)
    outcome = runtime.run(run_id, agent=agent)
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.result == {"answer": "ok"}
    assert runtime.store.get_run(run_id).repairs == 1
    runtime.close()
