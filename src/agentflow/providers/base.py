"""The interface every model provider implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from agentflow.messages import Message, ModelResponse


class ProviderError(Exception):
    """The provider failed in a way that retrying will not fix (bad key, bad request)."""


class TransientProviderError(ProviderError):
    """Rate limit, timeout or server error. The runtime retries these a bounded number of times.

    A model call has no side effect outside the provider, so retrying it is always safe; the only
    cost is tokens.
    """


class Provider(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        output_schema: dict[str, Any],
    ) -> ModelResponse:
        """Return the next assistant turn for ``messages``."""
        ...
