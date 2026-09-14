"""Thin MCP client wrapper.

One connection is opened per evaluation run and every tool call inside that run
shares it. Sessions are not held across Streamlit reruns on purpose -- session
lifetime bugs are the number one source of flakiness in tools like this.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@dataclass
class ServerConfig:
    """How to reach the MCP server under test."""

    transport: str = "http"            # "http" | "sse" | "stdio"
    url: str = "http://localhost:3000/mcp"
    command: str = ""                  # stdio only, e.g. "node dist/server.js"
    headers: dict[str, str] = field(default_factory=dict)

    def label(self) -> str:
        return self.url if self.transport != "stdio" else self.command


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        schema = self.schema or {"type": "object", "properties": {}}
        if "type" not in schema:
            schema = {**schema, "type": "object"}
        return {
            "name": self.name,
            "description": self.description or "",
            "input_schema": schema,
        }


@dataclass
class CallResult:
    ok: bool
    text: str
    latency_ms: int
    error: str = ""

    @property
    def approx_tokens(self) -> int:
        # Rough but consistent. Consistency is what matters for comparing runs.
        return max(1, len(self.text) // 4)


@asynccontextmanager
async def open_session(cfg: ServerConfig):
    """Yield a live, initialised ClientSession."""
    if cfg.transport == "stdio":
        parts = shlex.split(cfg.command)
        if not parts:
            raise ValueError("stdio transport needs a command")
        params = StdioServerParameters(command=parts[0], args=parts[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    elif cfg.transport == "sse":
        async with sse_client(cfg.url, headers=cfg.headers or None) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        async with streamable_http_client(cfg.url, headers=cfg.headers or None) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_tools(session: ClientSession) -> list[ToolSpec]:
    resp = await session.list_tools()
    return [
        ToolSpec(
            name=t.name,
            description=t.description or "",
            schema=(t.inputSchema or {}) if hasattr(t, "inputSchema") else {},
        )
        for t in resp.tools
    ]


async def call_tool(session: ClientSession, name: str, args: dict) -> CallResult:
    started = time.perf_counter()
    try:
        resp = await session.call_tool(name, args or {})
        elapsed = int((time.perf_counter() - started) * 1000)
        chunks: list[str] = []
        for block in resp.content or []:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
            else:
                chunks.append(json.dumps(getattr(block, "__dict__", {}), default=str))
        text = "\n".join(chunks)
        if getattr(resp, "isError", False):
            return CallResult(False, text, elapsed, error="tool reported error")
        return CallResult(True, text, elapsed)
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded
        elapsed = int((time.perf_counter() - started) * 1000)
        return CallResult(False, "", elapsed, error=f"{type(exc).__name__}: {exc}")


def tools_hash(tools: list[ToolSpec]) -> str:
    """Stable fingerprint of the exposed tool surface.

    Recorded on every run so you can always prove which server shape produced
    a given number.
    """
    payload = json.dumps(
        sorted([{"n": t.name, "d": t.description} for t in tools], key=lambda x: x["n"]),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
