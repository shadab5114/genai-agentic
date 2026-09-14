"""Evaluation runners.

R0  static   - no model, no intents. Reads the tool surface itself.
R1  direct   - no model. Calls tools straight with the intent text.
R2  routing  - model, single turn. Measures tool selection.
R3  task     - model, multi-turn. Measures the whole loop on build tasks.

R2 and R3 share one agent loop; they differ only in the system prompt and in
how many turns they are allowed.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from anthropic import AsyncAnthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import store
from .intents import Intent, IntentSet, tag_result
from .mcp_client import (
    CallResult,
    ServerConfig,
    ToolSpec,
    call_tool,
    list_tools,
    open_session,
    tools_hash,
)

Progress = Callable[[float, str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


# --------------------------------------------------------------------------
# R0 - static audit
# --------------------------------------------------------------------------

def _normalise(name: str, desc: str) -> str:
    """Split snake_case names into words and keep common words.

    Stop-word removal and bigrams both hurt here: tool descriptions are short and
    what makes two of them confusable is precisely the shared boilerplate
    ("...for the design system"). Stripping it hides the overlap you are hunting.
    """
    return (re.sub(r"[_\-]+", " ", name) + " " + (desc or "")).lower()


def static_audit(tools: list[ToolSpec], threshold: float = 0.45) -> dict[str, Any]:
    """Inspect the tool surface without calling anything.

    The similarity matrix predicts routing confusion before a single token is
    spent: if two descriptions read the same to a vectoriser, they read the same
    to a model choosing between them.
    """
    names = [t.name for t in tools]
    corpus = [_normalise(t.name, t.description) for t in tools]

    pairs: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    if len(tools) >= 2:
        vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=1)
        X = vec.fit_transform(corpus)
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)
        matrix = sim.round(3).tolist()
        for a, b in itertools.combinations(range(len(tools)), 2):
            if sim[a][b] >= threshold:
                pairs.append(
                    {"tool_a": names[a], "tool_b": names[b], "similarity": round(float(sim[a][b]), 3)}
                )
        pairs.sort(key=lambda p: -p["similarity"])

    BOUNDARY_HINTS = ("do not", "does not", "not for", "instead use", "rather than", "unlike")
    rows = []
    for t in tools:
        desc = t.description or ""
        props = (t.schema or {}).get("properties", {}) or {}
        required = (t.schema or {}).get("required", []) or []
        rows.append(
            {
                "tool": t.name,
                "desc_words": len(desc.split()),
                "has_description": bool(desc.strip()),
                "declares_boundary": any(h in desc.lower() for h in BOUNDARY_HINTS),
                "n_params": len(props),
                "n_required": len(required),
                "has_enum_param": any("enum" in (p or {}) for p in props.values()),
                "free_text_only": bool(props)
                and all((p or {}).get("type") == "string" and "enum" not in (p or {})
                        for p in props.values()),
            }
        )

    return {
        "generated_at": _now(),
        "n_tools": len(tools),
        "tools": rows,
        "similar_pairs": pairs,
        "names": names,
        "matrix": matrix,
        "threshold": threshold,
    }


async def probe_response_sizes(
    cfg: ServerConfig,
    tools: list[ToolSpec],
    sample_value: str,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """Call every single-string-parameter tool once to profile payload size.

    Tools needing richer arguments are skipped rather than guessed at.
    """
    out: list[dict[str, Any]] = []
    async with open_session(cfg) as session:
        for n, t in enumerate(tools):
            props = (t.schema or {}).get("properties", {}) or {}
            required = (t.schema or {}).get("required", []) or []
            if progress:
                progress((n + 1) / max(1, len(tools)), f"probing {t.name}")
            if len(required) > 1:
                out.append({"tool": t.name, "status": "skipped (multi-arg)"})
                continue
            args: dict[str, Any] = {}
            if required:
                key = required[0]
                if (props.get(key) or {}).get("type") not in (None, "string"):
                    out.append({"tool": t.name, "status": "skipped (non-string arg)"})
                    continue
                args = {key: sample_value}
            res = await call_tool(session, t.name, args)
            out.append(
                {
                    "tool": t.name,
                    "status": "ok" if res.ok else f"error: {res.error[:80]}",
                    "latency_ms": res.latency_ms,
                    "resp_tokens": res.approx_tokens if res.ok else 0,
                    "resp_chars": len(res.text),
                }
            )
    return out


# --------------------------------------------------------------------------
# R1 - direct tool probing (no model)
# --------------------------------------------------------------------------

async def run_direct(
    cfg: ServerConfig,
    intent_set: IntentSet,
    tools: list[ToolSpec],
    label: str = "",
    progress: Progress | None = None,
) -> str:
    """For each intent, call its expected tools directly and check for a hit.

    This isolates retrieval quality from routing entirely. If R1 passes and R2
    fails, the data is fine and the descriptions are the problem.
    """
    conn = store.connect()
    run_id = _run_id("R1")
    by_name = {t.name: t for t in tools}

    store.save_run(
        conn,
        {
            "run_id": run_id,
            "created_at": _now(),
            "label": label or "direct probe",
            "runner": "R1_direct",
            "model": "",
            "server_label": cfg.label(),
            "tools_hash": tools_hash(tools),
            "tools_enabled": json.dumps([t.name for t in tools]),
            "intent_set": intent_set.version,
            "repeats": 1,
            "notes": "no model in the loop",
        },
    )

    total = len(intent_set.intents)
    async with open_session(cfg) as session:
        for n, intent in enumerate(intent_set.intents):
            if progress:
                progress((n + 1) / total, f"{intent.id} {intent.text[:48]}")
            called, outputs = [], []
            t_ms = tok = 0
            for idx, tool_name in enumerate(intent.expected_tools):
                spec = by_name.get(tool_name)
                if spec is None:
                    continue
                args = _guess_args(spec, intent.text)
                res = await call_tool(session, tool_name, args)
                called.append(tool_name)
                outputs.append(res.text)
                t_ms += res.latency_ms
                tok += res.approx_tokens
                hit = any(k.lower() in res.text.lower() for k in intent.answer_contains)
                store.save_tool_call(
                    conn,
                    {
                        "run_id": run_id, "intent_id": intent.id, "repeat_idx": 0,
                        "call_idx": idx, "tool": tool_name, "args": json.dumps(args),
                        "ok": int(res.ok), "hit": int(hit), "latency_ms": res.latency_ms,
                        "resp_tokens": res.approx_tokens,
                        "resp_excerpt": res.text[:1500], "error": res.error,
                    },
                )
            joined = "\n".join(outputs)
            tag, resolved = tag_result(intent, called, outputs, joined)
            store.save_intent_result(
                conn,
                {
                    "run_id": run_id, "intent_id": intent.id, "repeat_idx": 0,
                    "shape": intent.shape, "stage": intent.stage,
                    "answerable": int(intent.answerable),
                    "expected_tools": json.dumps(intent.expected_tools),
                    "called_tools": json.dumps(called), "n_calls": len(called),
                    "tokens_tools": tok, "tokens_model_in": 0, "tokens_model_out": 0,
                    "latency_ms": t_ms, "resolved": int(resolved),
                    "first_correct": int(bool(called) and called[0] in intent.ok_tools),
                    "any_correct": int(bool(set(called) & intent.ok_tools)),
                    "stop_short": 0, "fished": 0, "zero_call": int(not called),
                    "tag": tag, "final_answer": joined[:2000],
                },
            )
            conn.commit()
    conn.close()
    return run_id


def _guess_args(spec: ToolSpec, text: str) -> dict[str, Any]:
    props = (spec.schema or {}).get("properties", {}) or {}
    required = (spec.schema or {}).get("required", []) or []
    args: dict[str, Any] = {}
    # Only ever send parameters the schema actually declares; servers reject
    # unknown keys and the resulting error looks like a data failure when it is not.
    keys = list(required) or [k for k in props if (props[k] or {}).get("type") == "string"][:1]
    for key in keys:
        if key not in props and props:
            continue
        prop = props.get(key) or {}
        if prop.get("enum"):
            args[key] = prop["enum"][0]
        elif prop.get("type") == "string":
            args[key] = text
        elif prop.get("type") in ("integer", "number"):
            args[key] = 1
        elif prop.get("type") == "boolean":
            args[key] = True
        else:
            args[key] = text
    return args


# --------------------------------------------------------------------------
# R2 / R3 - agent loop
# --------------------------------------------------------------------------

ROUTING_PROMPT = (
    "You are a developer working in a codebase that uses the Verizon Design System. "
    "Answer the question using the available tools. Call tools as needed, then give "
    "a short, direct answer. Do not speculate: if the tools do not give you the "
    "answer, say so plainly."
)

TASK_PROMPT = (
    "You are a developer building UI with the Verizon Design System. Use the "
    "available tools to discover the right components and their correct APIs, then "
    "produce the JSX. Do not invent component or prop names."
)


async def run_agent(
    cfg: ServerConfig,
    intent_set: IntentSet,
    tools: list[ToolSpec],
    model: str,
    api_key: str,
    runner: str = "R2_routing",
    repeats: int = 3,
    max_turns: int = 6,
    enabled_tools: list[str] | None = None,
    label: str = "",
    notes: str = "",
    progress: Progress | None = None,
) -> str:
    """Run every intent through a real model with the MCP tools attached."""
    conn = store.connect()
    run_id = _run_id(runner.split("_")[0])
    client = AsyncAnthropic(api_key=api_key)

    active = [t for t in tools if enabled_tools is None or t.name in enabled_tools]
    schemas = [t.to_anthropic() for t in active]
    system = TASK_PROMPT if runner.startswith("R3") else ROUTING_PROMPT

    store.save_run(
        conn,
        {
            "run_id": run_id, "created_at": _now(),
            "label": label or runner, "runner": runner, "model": model,
            "server_label": cfg.label(), "tools_hash": tools_hash(active),
            "tools_enabled": json.dumps([t.name for t in active]),
            "intent_set": intent_set.version, "repeats": repeats, "notes": notes,
        },
    )

    total = len(intent_set.intents) * repeats
    done = 0
    async with open_session(cfg) as session:
        for intent in intent_set.intents:
            for rep in range(repeats):
                done += 1
                if progress:
                    progress(done / total, f"[{rep + 1}/{repeats}] {intent.id} {intent.text[:42]}")
                try:
                    result = await _agent_once(
                        client, session, model, system, schemas, intent, max_turns
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "called": [], "outputs": [], "answer": f"HARNESS ERROR: {exc}",
                        "calls": [], "latency_ms": 0, "tok_in": 0, "tok_out": 0,
                    }

                for c in result["calls"]:
                    store.save_tool_call(
                        conn,
                        {
                            "run_id": run_id, "intent_id": intent.id, "repeat_idx": rep,
                            "call_idx": c["idx"], "tool": c["tool"],
                            "args": json.dumps(c["args"])[:1000], "ok": int(c["ok"]),
                            "hit": int(any(k.lower() in c["text"].lower()
                                           for k in intent.answer_contains)),
                            "latency_ms": c["latency_ms"], "resp_tokens": c["tokens"],
                            "resp_excerpt": c["text"][:1500], "error": c["error"],
                        },
                    )

                called = result["called"]
                tag, resolved = tag_result(intent, called, result["outputs"], result["answer"])
                expected = set(intent.expected_tools)
                counts = {t: called.count(t) for t in set(called)}
                store.save_intent_result(
                    conn,
                    {
                        "run_id": run_id, "intent_id": intent.id, "repeat_idx": rep,
                        "shape": intent.shape, "stage": intent.stage,
                        "answerable": int(intent.answerable),
                        "expected_tools": json.dumps(intent.expected_tools),
                        "called_tools": json.dumps(called), "n_calls": len(called),
                        "tokens_tools": sum(c["tokens"] for c in result["calls"]),
                        "tokens_model_in": result["tok_in"],
                        "tokens_model_out": result["tok_out"],
                        "latency_ms": result["latency_ms"], "resolved": int(resolved),
                        "first_correct": int(bool(called) and called[0] in intent.ok_tools),
                        "any_correct": int(bool(set(called) & intent.ok_tools)),
                        # stop_short: touched the area but missed a tool it needed
                        "stop_short": int(
                            bool(expected) and bool(set(called) & intent.ok_tools)
                            and not expected.issubset(set(called))
                        ),
                        # fished: same tool hammered with rephrasings -> no right entry point
                        "fished": int(any(v >= 3 for v in counts.values())),
                        "zero_call": int(not called),
                        "tag": tag, "final_answer": result["answer"][:4000],
                    },
                )
                conn.commit()
    conn.close()
    return run_id


async def _agent_once(client, session, model, system, schemas, intent: Intent, max_turns: int):
    messages: list[dict[str, Any]] = [{"role": "user", "content": intent.text}]
    called: list[str] = []
    outputs: list[str] = []
    calls: list[dict[str, Any]] = []
    tok_in = tok_out = 0
    idx = 0
    started = time.perf_counter()

    for _ in range(max_turns):
        resp = await client.messages.create(
            model=model, max_tokens=1500, system=system,
            tools=schemas, messages=messages,
        )
        tok_in += getattr(resp.usage, "input_tokens", 0) or 0
        tok_out += getattr(resp.usage, "output_tokens", 0) or 0

        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            text = "".join(getattr(b, "text", "") for b in resp.content)
            return {
                "called": called, "outputs": outputs, "answer": text, "calls": calls,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "tok_in": tok_in, "tok_out": tok_out,
            }

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for use in uses:
            res: CallResult = await call_tool(session, use.name, dict(use.input or {}))
            called.append(use.name)
            outputs.append(res.text)
            calls.append(
                {
                    "idx": idx, "tool": use.name, "args": dict(use.input or {}),
                    "ok": res.ok, "text": res.text, "tokens": res.approx_tokens,
                    "latency_ms": res.latency_ms, "error": res.error,
                }
            )
            idx += 1
            results.append(
                {
                    "type": "tool_result", "tool_use_id": use.id,
                    "content": res.text[:8000] if res.ok else f"ERROR: {res.error}",
                    "is_error": not res.ok,
                }
            )
        messages.append({"role": "user", "content": results})

    return {
        "called": called, "outputs": outputs,
        "answer": "(max turns reached without a final answer)", "calls": calls,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "tok_in": tok_in, "tok_out": tok_out,
    }


def sync(coro):
    """Run an async runner from Streamlit's synchronous context."""
    return asyncio.run(coro)
