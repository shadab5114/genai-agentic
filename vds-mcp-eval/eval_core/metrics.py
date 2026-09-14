"""Metrics computed from stored traces.

Nothing is computed at run time. Every number here is derived from the trace
store, so re-scoring an old run against a new rule costs nothing.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd


def intent_frame(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM intent_results WHERE run_id = ?", conn, params=(run_id,)
    )
    if df.empty:
        return df
    df["expected_tools"] = df["expected_tools"].apply(json.loads)
    df["called_tools"] = df["called_tools"].apply(json.loads)
    return df


def call_frame(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM tool_calls WHERE run_id = ?", conn, params=(run_id,)
    )


def headline(df: pd.DataFrame) -> dict[str, float]:
    """The five numbers that go at the top of the report."""
    if df.empty:
        return {}
    n = len(df)
    return {
        "Resolution rate": 100 * df["resolved"].mean(),
        "First-call accuracy": 100 * df["first_correct"].mean(),
        "Tool selection accuracy": 100 * df["any_correct"].mean(),
        "Stop-short rate": 100 * df["stop_short"].mean(),
        "Zero-call rate": 100 * df["zero_call"].mean(),
        "Fishing rate": 100 * df["fished"].mean(),
        "Mean calls per intent": df["n_calls"].mean(),
        "Median tool tokens per intent": df["tokens_tools"].median(),
        "p95 tool tokens per intent": df["tokens_tools"].quantile(0.95),
        "Median latency (ms)": df["latency_ms"].median(),
        "Intents evaluated": n,
    }


def stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-intent variance across repeats.

    High spread means the intent is a coin flip, which is itself a finding:
    unstable routing usually points at two tools the model cannot separate.
    """
    if df.empty or df["repeat_idx"].nunique() < 2:
        return pd.DataFrame()
    g = df.groupby("intent_id")["resolved"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std().fillna(0), "n": g.count()})
    out["unstable"] = (out["mean"] > 0) & (out["mean"] < 1)
    return out.sort_values(["unstable", "std"], ascending=False).reset_index()


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Expected tool (rows) against the tool actually called first (columns)."""
    rows = []
    for _, r in df.iterrows():
        exp = r["expected_tools"][0] if r["expected_tools"] else "(none)"
        got = r["called_tools"][0] if r["called_tools"] else "(no call)"
        rows.append({"expected": exp, "called_first": got})
    if not rows:
        return pd.DataFrame()
    cm = pd.crosstab(
        pd.DataFrame(rows)["expected"], pd.DataFrame(rows)["called_first"]
    )
    return cm


def tool_utilisation(
    calls: pd.DataFrame, all_tools: list[str], n_intents: int
) -> pd.DataFrame:
    """Call share, hit rate, cost and latency per tool -- including dead tools."""
    base = pd.DataFrame({"tool": all_tools})
    if calls.empty:
        base["calls"] = 0
        base["share_pct"] = 0.0
        base["hit_rate_pct"] = 0.0
        base["median_tokens"] = 0
        base["p95_tokens"] = 0
        base["median_latency_ms"] = 0
        base["error_rate_pct"] = 0.0
        base["dead"] = True
        return base

    g = calls.groupby("tool")
    agg = pd.DataFrame(
        {
            "calls": g.size(),
            "hit_rate_pct": 100 * g["hit"].mean(),
            "median_tokens": g["resp_tokens"].median(),
            "p95_tokens": g["resp_tokens"].quantile(0.95),
            "median_latency_ms": g["latency_ms"].median(),
            "error_rate_pct": 100 * (1 - g["ok"].mean()),
        }
    ).reset_index()

    out = base.merge(agg, on="tool", how="left").fillna(0)
    total = out["calls"].sum() or 1
    out["share_pct"] = (100 * out["calls"] / total).round(2)
    out["calls_per_intent"] = (out["calls"] / max(1, n_intents)).round(2)
    out["dead"] = out["calls"] == 0
    cols = [
        "tool", "calls", "share_pct", "calls_per_intent", "hit_rate_pct",
        "median_tokens", "p95_tokens", "median_latency_ms", "error_rate_pct", "dead",
    ]
    return out[cols].sort_values("calls", ascending=False).reset_index(drop=True)


def by_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage and accuracy per workflow stage -- the positioning answer."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("stage")
    out = pd.DataFrame(
        {
            "intents": g.size(),
            "answerable_pct": 100 * g["answerable"].mean(),
            "resolved_pct": 100 * g["resolved"].mean(),
            "selection_pct": 100 * g["any_correct"].mean(),
            "zero_call_pct": 100 * g["zero_call"].mean(),
            "mean_calls": g["n_calls"].mean(),
        }
    ).round(1)
    order = ["discover", "decide", "specify", "compose", "validate", "meta"]
    out = out.reindex([s for s in order if s in out.index])
    return out.reset_index()


def by_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Adopter-phrased versus agent-harvested questions."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("shape")
    return pd.DataFrame(
        {
            "intents": g.size(),
            "resolved_pct": (100 * g["resolved"].mean()).round(1),
            "selection_pct": (100 * g["any_correct"].mean()).round(1),
            "mean_calls": g["n_calls"].mean().round(2),
            "median_tokens": g["tokens_tools"].median(),
        }
    ).reset_index()


def tag_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    counts = df["tag"].value_counts().reset_index()
    counts.columns = ["tag", "count"]
    counts["pct"] = (100 * counts["count"] / len(df)).round(1)
    return counts


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Side-by-side deltas. This is the before/after table."""
    ha, hb = headline(a), headline(b)
    keys = [k for k in ha if k in hb]
    rows = []
    for k in keys:
        delta = hb[k] - ha[k]
        rows.append(
            {
                "metric": k,
                label_a: round(ha[k], 2),
                label_b: round(hb[k], 2),
                "delta": round(delta, 2),
            }
        )
    return pd.DataFrame(rows)


def ablation_table(conn: sqlite3.Connection, baseline_run: str, ablation_runs: dict[str, str]) -> pd.DataFrame:
    """Marginal value of each tool: how much resolution drops when it is removed.

    Zero or negative marginal value is the evidence for merging or deleting a
    tool. Negative means the tool was actively confusing the model.
    """
    base = intent_frame(conn, baseline_run)
    if base.empty:
        return pd.DataFrame()
    base_rate = 100 * base["resolved"].mean()
    rows = []
    for tool, run_id in ablation_runs.items():
        df = intent_frame(conn, run_id)
        if df.empty:
            continue
        rate = 100 * df["resolved"].mean()
        rows.append(
            {
                "tool removed": tool,
                "resolution without it": round(rate, 2),
                "baseline": round(base_rate, 2),
                "marginal value": round(base_rate - rate, 2),
            }
        )
    out = pd.DataFrame(rows).sort_values("marginal value")
    if not out.empty:
        out["verdict"] = out["marginal value"].apply(
            lambda v: "remove or merge" if v <= 0.0 else ("low value" if v < 1.0 else "earning its place")
        )
    return out.reset_index(drop=True)


def recommendations(
    df: pd.DataFrame, utilisation: pd.DataFrame, stages: pd.DataFrame, audit: dict | None
) -> list[dict[str, str]]:
    """Turn the numbers into a ranked list of changes, with the evidence attached."""
    recs: list[dict[str, str]] = []
    if df.empty:
        return recs

    tags = df["tag"].value_counts(normalize=True) * 100

    if tags.get("R", 0) >= 20:
        recs.append(
            {
                "priority": "1",
                "area": "Tool descriptions",
                "finding": f"{tags['R']:.0f}% of intents failed on routing (tag R).",
                "action": "Rewrite descriptions and declare boundaries before touching any documentation.",
            }
        )
    if tags.get("C", 0) >= 10:
        recs.append(
            {
                "priority": "1",
                "area": "Coverage",
                "finding": f"{tags['C']:.0f}% of intents have no tool that can answer them.",
                "action": "Add a tool for the uncovered stages listed in the coverage table.",
            }
        )
    if tags.get("P", 0) >= 10:
        recs.append(
            {
                "priority": "2",
                "area": "Response shape",
                "finding": f"{tags['P']:.0f}% of intents had the answer in a tool response but not in the final answer.",
                "action": "Shrink payloads and lead with the answer; the agent is losing it in bulk.",
            }
        )
    if tags.get("D", 0) >= 10:
        recs.append(
            {
                "priority": "2",
                "area": "Documentation",
                "finding": f"{tags['D']:.0f}% of intents reached the right tool but the content was insufficient.",
                "action": "This is a docs problem, not a server problem. Route it to the content backlog.",
            }
        )
    if tags.get("Q", 0) >= 10:
        recs.append(
            {
                "priority": "2",
                "area": "Search and synonyms",
                "finding": f"{tags['Q']:.0f}% of intents hit the right tool but matched nothing.",
                "action": "Add synonyms and intent aliases to the search index.",
            }
        )

    dead = utilisation[utilisation["dead"]]["tool"].tolist() if not utilisation.empty else []
    if dead:
        recs.append(
            {
                "priority": "2",
                "area": "Tool surface",
                "finding": f"{len(dead)} tools were never called: {', '.join(dead[:8])}"
                + ("…" if len(dead) > 8 else ""),
                "action": "Confirm with leave-one-out ablation, then merge or remove.",
            }
        )

    if not utilisation.empty:
        hogs = utilisation[utilisation["p95_tokens"] > 1500]["tool"].tolist()
        if hogs:
            recs.append(
                {
                    "priority": "3",
                    "area": "Payload size",
                    "finding": f"{len(hogs)} tools have a p95 response over 1500 tokens: {', '.join(hogs[:6])}.",
                    "action": "Add a summary mode or paginate; these are eating the task's context budget.",
                }
            )

    if audit and audit.get("similar_pairs"):
        top = audit["similar_pairs"][0]
        recs.append(
            {
                "priority": "1",
                "area": "Description overlap",
                "finding": f"{len(audit['similar_pairs'])} tool pairs are near-duplicates by description; "
                f"worst is {top['tool_a']} vs {top['tool_b']} at {top['similarity']}.",
                "action": "Merge, or rewrite each description to state what the other one covers.",
            }
        )

    if not stages.empty:
        weak = stages[stages["resolved_pct"] < 50]["stage"].tolist()
        if weak:
            recs.append(
                {
                    "priority": "1",
                    "area": "Tool positioning",
                    "finding": f"Resolution is under 50% at these workflow stages: {', '.join(weak)}.",
                    "action": "These stages have no effective entry point. Add or reposition a tool.",
                }
            )

    zero = 100 * df["zero_call"].mean()
    if zero >= 15:
        recs.append(
            {
                "priority": "1",
                "area": "Tool discoverability",
                "finding": f"The agent answered without calling any tool on {zero:.0f}% of intents.",
                "action": "Descriptions are not signalling relevance; the model is trusting its priors instead.",
            }
        )

    return sorted(recs, key=lambda r: r["priority"])
