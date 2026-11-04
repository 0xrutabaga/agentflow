"""Agent definitions and how a resumed process finds them again.

A run stores an ``agent_ref`` such as ``agentflow.examples.triage:build_agent``. Resuming in a
new process imports that factory and calls it with the data directory, so the tool registry,
provider and output schema are rebuilt from code, never deserialised from the database.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from agentflow.providers.base import Provider
from agentflow.schema import make_validator
from agentflow.tools import ToolRegistry
from agentflow.trace import Redactor, redact_sensitive_keys


@dataclass(frozen=True)
class Limits:
    """Per-run budgets. ``max_wall_s`` becomes an absolute deadline stored with the run, so it
    keeps counting across crashes and resumes."""

    max_steps: int = 20  # model calls
    max_tool_calls: int = 30  # tool calls requested by the model (rejected ones included)
    max_repairs: int = 2  # invalid outputs or invalid tool calls fed back for repair
    max_wall_s: float | None = 300.0
    model_retries: int = 2  # extra attempts after a TransientProviderError


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    tools: ToolRegistry
    provider: Provider
    output_schema: dict[str, Any]
    limits: Limits = field(default_factory=Limits)
    redactor: Redactor = redact_sensitive_keys
    output_validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_validator", make_validator(self.output_schema))


AgentFactory = Callable[[Path], Agent]


def load_agent(ref: str, data_dir: Path) -> Agent:
    """Import ``module:function`` and call it with ``data_dir``."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"agent reference must look like 'package.module:factory', got {ref!r}")
    factory = getattr(importlib.import_module(module_name), attr)
    agent = factory(data_dir)
    if not isinstance(agent, Agent):
        raise TypeError(f"{ref} returned {type(agent).__name__}, expected Agent")
    return agent
