"""Selection probes — measuring how the model reads your tool surface.

At selection time a model sees names, descriptions and schemas. Nothing else.
So tool organisation can be measured with no MCP connection, no tool execution
and no latency: just the tool list and a question.

R2 records which tool was chosen. These probes record how the choice was made --
the runner-up, the stated reason, the confidence, and how stable the pick is
across samples. Same failure, very different fixes.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from typing import Any, Callable

import pandas as pd
from anthropic import AsyncAnthropic

from .intents import Intent, IntentSet
from .mcp_client import ToolSpec

Progress = Callable[[float, str], None]

PROBE_SYSTEM = """You are choosing which tool to call to answer a developer's question about a design system.

You are given a list of tools. Do NOT call anything. Decide which tool you would call.

Return ONLY a JSON object, no prose and no markdown fences:
{"ranked": [{"tool": "<exact name>", "confidence": <0.0-1.0>, "reason": "<one short line>"}]}

Rules:
- Rank your top 3 candidates, best first. If only one is plausible, return one.
- confidence is your honest probability that this is the right tool, not a rank score.
- If no tool fits, return {"ranked": [{"tool": "NONE", "confidence": 1.0, "reason": "..."}]}.
- In reason, say what in the tool's description drove the choice. If two tools look
  interchangeable to you, say so explicitly."""


def _render_tools(tools: list[ToolSpec]) -> str:
    lines = []
    for t in tools:
        params = list(((t.schema or {}).get("properties", {}) or {}).keys())
        p = f"  params: {', '.join(params)}" if params else "  params: none"
        lines.append(f"- {t.name}\n  {t.description or '(no description)'}\n{p}")
    return "\n".join(lines)


async def _ask_once(client, model, tools: list[ToolSpec], question: str, temperature: float):
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        temperature=temperature,
        system=PROBE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"TOOLS:\n{_render_tools(tools)}\n\nQUESTION:\n{question}",
            }
        ],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        ranked = data.get("ranked", [])
        return [
            {
                "tool": str(r.get("tool", "")),
                "confidence": float(r.get("confidence", 0) or 0),
                "reason": str(r.get("reason", "")),
            }
            for r in ranked
            if r.get("tool")
        ]
    except Exception:  # noqa: BLE001
        return []


def _entropy(picks: list[str]) -> float:
    """Normalised Shannon entropy of the top-pick distribution across samples.

    0.0 means the model always picks the same tool. 1.0 means it is spreading
    evenly across candidates, which is what an ambiguous surface looks like.
    """
    if len(picks) < 2:
        return 0.0
    counts = Counter(picks)
    n = len(picks)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return round(h / math.log(min(n, len(counts)) or 1, math.e), 3) if len(counts) > 1 else 0.0


async def forced_choice(
    intent_set: IntentSet,
    tools: list[ToolSpec],
    model: str,
    api_key: str,
    samples: int = 5,
    temperature: float = 1.0,
    shuffle_order: bool = True,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Ask the model to rank tools for each intent, several times.

    shuffle_order re-orders the tool list between samples. If accuracy depends on
    ordering, the descriptions are not carrying the decision.
    """
    client = AsyncAnthropic(api_key=api_key)
    rows: list[dict[str, Any]] = []
    total = len(intent_set.intents)

    for n, intent in enumerate(intent_set.intents):
        if progress:
            progress((n + 1) / total, f"{intent.id} {intent.text[:48]}")

        samples_out = []
        for s in range(samples):
            listing = list(tools)
            if shuffle_order:
                random.Random(s).shuffle(listing)
            samples_out.append(await _ask_once(client, model, listing, intent.text, temperature))

        tops = [r[0]["tool"] for r in samples_out if r]
        ok = intent.ok_tools

        ranks = []
        for r in samples_out:
            names = [c["tool"] for c in r]
            hit = next((i + 1 for i, nm in enumerate(names) if nm in ok), None)
            ranks.append(hit)
        found = [r for r in ranks if r]

        top_conf = [r[0]["confidence"] for r in samples_out if r]
        reasons = [c["reason"] for r in samples_out for c in r[:1] if c.get("reason")]
        runners_up = [c["tool"] for r in samples_out for c in r[1:2]]

        top1 = sum(1 for t in tops if t in ok) / max(1, len(tops))
        top3 = sum(1 for r in ranks if r) / max(1, len(ranks))
        mean_conf = sum(top_conf) / len(top_conf) if top_conf else 0.0

        rows.append(
            {
                "intent_id": intent.id,
                "text": intent.text,
                "stage": intent.stage,
                "shape": intent.shape,
                "expected": ", ".join(sorted(ok)) or "(none)",
                "top_pick": Counter(tops).most_common(1)[0][0] if tops else "(none)",
                "top1_acc": round(100 * top1, 1),
                "top3_acc": round(100 * top3, 1),
                "mean_rank_of_correct": round(sum(found) / len(found), 2) if found else None,
                "mean_confidence": round(mean_conf, 2),
                # The dangerous quadrant: sure of itself, and wrong.
                "confidently_wrong": bool(top1 < 0.5 and mean_conf >= 0.6),
                "entropy": _entropy(tops),
                "runner_up": Counter(runners_up).most_common(1)[0][0] if runners_up else "",
                "reason": reasons[0] if reasons else "",
                "all_reasons": " | ".join(dict.fromkeys(reasons))[:600],
                "pick_spread": ", ".join(f"{k}×{v}" for k, v in Counter(tops).most_common()),
            }
        )
    return pd.DataFrame(rows)


async def pairwise_discrimination(
    tool_a: ToolSpec,
    tool_b: ToolSpec,
    intents: list[Intent],
    model: str,
    api_key: str,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Present only two tools and see whether the model can tell them apart.

    If it cannot separate them head to head, it will never separate them among
    thirty. This is the cleanest merge argument available.
    """
    client = AsyncAnthropic(api_key=api_key)
    pair = [tool_a, tool_b]
    relevant = [i for i in intents if i.ok_tools & {tool_a.name, tool_b.name}]
    rows = []
    for n, intent in enumerate(relevant):
        if progress:
            progress((n + 1) / max(1, len(relevant)), intent.id)
        ranked = await _ask_once(client, model, pair, intent.text, 0.0)
        pick = ranked[0]["tool"] if ranked else "(none)"
        rows.append(
            {
                "intent_id": intent.id,
                "text": intent.text[:70],
                "expected": ", ".join(sorted(intent.ok_tools & {tool_a.name, tool_b.name})),
                "picked": pick,
                "correct": pick in intent.ok_tools,
                "reason": ranked[0]["reason"] if ranked else "",
            }
        )
    df = pd.DataFrame(rows)
    acc = 100 * df["correct"].mean() if not df.empty else 0.0
    return {
        "pair": f"{tool_a.name} vs {tool_b.name}",
        "n": len(df),
        "accuracy": round(acc, 1),
        # 50% on a two-way choice is a coin flip: the tools are indistinguishable.
        "verdict": (
            "indistinguishable — merge" if acc < 60
            else "weak separation — rewrite descriptions" if acc < 85
            else "separable — keep both"
        ),
        "detail": df,
    }


async def scaling_curve(
    intent_set: IntentSet,
    tools: list[ToolSpec],
    model: str,
    api_key: str,
    sizes: list[int] | None = None,
    seed: int = 0,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Accuracy against the number of tools in the list.

    Every list contains the correct tool plus random distractors, so the only
    variable is how many other options the model is weighing. A downward slope
    means the tool count itself is costing you accuracy.
    """
    client = AsyncAnthropic(api_key=api_key)
    sizes = sorted(set(sizes or [5, 10, 20, len(tools)]))
    by_name = {t.name: t for t in tools}
    rng = random.Random(seed)
    rows = []
    total = len(sizes) * len(intent_set.intents)
    done = 0

    for size in sizes:
        correct = 0
        counted = 0
        for intent in intent_set.intents:
            done += 1
            if progress:
                progress(done / total, f"{size} tools · {intent.id}")
            ok = [n for n in intent.ok_tools if n in by_name]
            if not ok:
                continue
            keeper = by_name[ok[0]]
            others = [t for t in tools if t.name != keeper.name]
            rng.shuffle(others)
            listing = [keeper] + others[: max(0, size - 1)]
            rng.shuffle(listing)
            ranked = await _ask_once(client, model, listing, intent.text, 0.0)
            pick = ranked[0]["tool"] if ranked else ""
            counted += 1
            correct += int(pick in intent.ok_tools)
        rows.append(
            {
                "tools_in_list": size,
                "intents": counted,
                "top1_accuracy": round(100 * correct / counted, 1) if counted else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sync(coro):
    return asyncio.run(coro)
