"""VDS MCP Eval — a harness for measuring an MCP server's behaviour.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from eval_core import metrics, probes, store
from eval_core.intents import (
    TAG_MEANING,
    available_sets,
    load_intent_set,
    validate,
)
from eval_core.mcp_client import ServerConfig, list_tools, open_session, tools_hash
from eval_core.runners import (
    probe_response_sizes,
    run_agent,
    run_direct,
    static_audit,
    sync,
)

st.set_page_config(page_title="VDS MCP Eval", layout="wide")
conn = store.connect()

# ---------------------------------------------------------------- sidebar

st.sidebar.title("VDS MCP Eval")
st.sidebar.caption("Measure the server. Prove the change.")

transport = st.sidebar.selectbox("Transport", ["http", "sse", "stdio"], index=0)
url = st.sidebar.text_input("Server URL", os.getenv("MCP_URL", "http://localhost:3000/mcp"))
command = st.sidebar.text_input("stdio command", os.getenv("MCP_CMD", ""))
header_raw = st.sidebar.text_area("Headers (JSON)", "{}", height=68)
try:
    headers = json.loads(header_raw or "{}")
except json.JSONDecodeError:
    headers = {}
    st.sidebar.error("Headers must be valid JSON")

cfg = ServerConfig(transport=transport, url=url, command=command, headers=headers)

st.sidebar.divider()
model = st.sidebar.text_input("Model (pin this)", os.getenv("EVAL_MODEL", "claude-sonnet-4-5"))
api_key = st.sidebar.text_input(
    "Anthropic API key", os.getenv("ANTHROPIC_API_KEY", ""), type="password"
)

if st.sidebar.button("Connect / refresh tools", use_container_width=True):
    try:
        async def _fetch():
            async with open_session(cfg) as s:
                return await list_tools(s)

        tools = sync(_fetch())
        st.session_state["tools"] = [
            {"name": t.name, "description": t.description, "schema": t.schema} for t in tools
        ]
        st.sidebar.success(f"{len(tools)} tools found")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"{type(exc).__name__}: {exc}")


def current_tools():
    from eval_core.mcp_client import ToolSpec

    return [ToolSpec(**t) for t in st.session_state.get("tools", [])]


tools = current_tools()
tool_names = {t.name for t in tools}

(
    tab_server, tab_intents, tab_run, tab_probe, tab_report, tab_compare
) = st.tabs(
    ["Server & audit", "Intents", "Run", "Selection probe", "Report", "Compare"]
)

# ---------------------------------------------------------------- server

with tab_server:
    st.header("Tool surface")
    if not tools:
        st.info("Connect to the server from the sidebar to begin.")
    else:
        st.caption(f"Fingerprint `{tools_hash(tools)}` — recorded on every run.")
        audit = static_audit(tools)
        st.session_state["audit"] = audit

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tools exposed", audit["n_tools"])
        c2.metric("Near-duplicate pairs", len(audit["similar_pairs"]))
        no_boundary = sum(1 for r in audit["tools"] if not r["declares_boundary"])
        c3.metric("Without boundaries", no_boundary)
        free_text = sum(1 for r in audit["tools"] if r["free_text_only"])
        c4.metric("Free-text params only", free_text)

        st.subheader("Overlapping descriptions")
        st.caption(
            "If a vectoriser cannot separate two descriptions, a model choosing "
            "between them cannot either. These pairs predict your confusion matrix."
        )
        if audit["similar_pairs"]:
            st.dataframe(pd.DataFrame(audit["similar_pairs"]), use_container_width=True, hide_index=True)
        else:
            st.success("No pairs above threshold.")

        st.subheader("Per-tool schema quality")
        st.dataframe(pd.DataFrame(audit["tools"]), use_container_width=True, hide_index=True)

        st.subheader("Response size probe")
        st.caption("Calls every single-string-argument tool once to profile payload size.")
        sample = st.text_input("Sample argument value", "Button")
        if st.button("Run probe"):
            bar = st.progress(0.0, text="starting")
            rows = sync(
                probe_response_sizes(
                    cfg, tools, sample, lambda p, m: bar.progress(min(p, 1.0), text=m)
                )
            )
            bar.empty()
            st.session_state["probe"] = rows
        if "probe" in st.session_state:
            pdf = pd.DataFrame(st.session_state["probe"])
            st.dataframe(pdf, use_container_width=True, hide_index=True)
            if "resp_tokens" in pdf:
                big = pdf[pdf["resp_tokens"] > 1500]
                if not big.empty:
                    st.warning(
                        f"{len(big)} tools return more than 1500 tokens for a single call. "
                        "At six components that is most of a context window."
                    )

# ---------------------------------------------------------------- intents

with tab_intents:
    st.header("Intent set")
    sets = available_sets()
    if not sets:
        st.error("No intent files found in ./intents")
    else:
        choice = st.selectbox("File", [p.name for p in sets])
        iset = load_intent_set([p for p in sets if p.name == choice][0])
        st.session_state["intent_set"] = iset

        st.caption(f"Version `{iset.version}` — {len(iset.intents)} intents. Freeze this file before your baseline run.")

        problems = validate(iset, tool_names)
        if problems:
            st.warning("Validation issues:\n\n" + "\n".join(f"- {p}" for p in problems[:25]))
        else:
            st.success("Intent set validates cleanly.")

        df = pd.DataFrame([vars(i) for i in iset.intents])
        c1, c2, c3 = st.columns(3)
        c1.metric("Intents", len(df))
        c2.metric("Answerable today", f"{100 * df['answerable'].mean():.0f}%")
        c3.metric("Agent-shaped", f"{100 * (df['shape'] == 'agent').mean():.0f}%")

        left, right = st.columns(2)
        left.caption("By stage")
        left.bar_chart(df["stage"].value_counts())
        right.caption("Coverage gap by stage (unanswerable)")
        gap = df[~df["answerable"]]["stage"].value_counts()
        right.bar_chart(gap if not gap.empty else pd.Series(dtype=int))

        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- run

with tab_run:
    st.header("Run an evaluation")
    iset = st.session_state.get("intent_set")
    if not tools or iset is None:
        st.info("Connect to a server and load an intent set first.")
    else:
        runner = st.radio(
            "Runner",
            ["R1 direct (no model)", "R2 routing (model, single turn)", "R3 task (model, multi-turn)"],
            horizontal=True,
        )
        label = st.text_input("Run label", "v1 baseline")
        notes = st.text_input("Notes", "current 33-tool surface, unchanged")

        if runner.startswith("R1"):
            st.caption(
                "Calls each intent's expected tools directly. No model, no cost. "
                "If R1 passes and R2 fails, your data is fine and the descriptions are not."
            )
            if st.button("Run R1", type="primary"):
                bar = st.progress(0.0, text="starting")
                rid = sync(
                    run_direct(cfg, iset, tools, label, lambda p, m: bar.progress(min(p, 1.0), text=m))
                )
                bar.empty()
                st.success(f"Run complete: `{rid}`")
        else:
            c1, c2, c3 = st.columns(3)
            repeats = c1.number_input("Repeats per intent", 1, 5, 3)
            max_turns = c2.number_input("Max turns", 1, 12, 6)
            subset = c3.number_input("Limit intents (0 = all)", 0, 500, 0)

            st.caption("Leave-one-out ablation re-runs the whole set once per removed tool. Expensive — start with a few suspects.")
            ablate = st.multiselect("Ablate these tools (optional)", sorted(tool_names))

            if not api_key:
                st.warning("An Anthropic API key is required for R2 and R3.")

            if st.button("Run", type="primary", disabled=not api_key):
                run_set = iset
                if subset:
                    from eval_core.intents import IntentSet

                    run_set = IntentSet(iset.version, iset.intents[:subset], iset.path)

                kind = "R2_routing" if runner.startswith("R2") else "R3_task"
                bar = st.progress(0.0, text="starting")
                rid = sync(
                    run_agent(
                        cfg, run_set, tools, model, api_key, kind, int(repeats),
                        int(max_turns), None, label, notes,
                        lambda p, m: bar.progress(min(p, 1.0), text=m),
                    )
                )
                bar.empty()
                st.success(f"Baseline run: `{rid}`")

                ablation_ids = {}
                for tool in ablate:
                    keep = [t.name for t in tools if t.name != tool]
                    b2 = st.progress(0.0, text=f"ablating {tool}")
                    ablation_ids[tool] = sync(
                        run_agent(
                            cfg, run_set, tools, model, api_key, kind, 1,
                            int(max_turns), keep, f"{label} — no {tool}",
                            f"leave-one-out: {tool}",
                            lambda p, m, t=tool: b2.progress(min(p, 1.0), text=f"{t}: {m}"),
                        )
                    )
                    b2.empty()
                if ablation_ids:
                    st.session_state["ablation"] = {"baseline": rid, "runs": ablation_ids}
                    st.success(f"Ablated {len(ablation_ids)} tools.")

        st.divider()
        st.subheader("Stored runs")
        runs = store.list_runs(conn)
        if runs:
            st.dataframe(
                pd.DataFrame([dict(r) for r in runs])[
                    ["run_id", "created_at", "label", "runner", "model", "tools_hash", "repeats"]
                ],
                use_container_width=True, hide_index=True,
            )
            drop = st.selectbox("Delete a run", ["—"] + [r["run_id"] for r in runs])
            if drop != "—" and st.button("Delete", type="secondary"):
                store.delete_run(conn, drop)
                st.rerun()

# ---------------------------------------------------------------- probe

with tab_probe:
    st.header("How the model reads your tool surface")
    st.caption(
        "No MCP connection is used here. At selection time the model sees only names, "
        "descriptions and schemas, so tool organisation can be measured without "
        "executing anything."
    )
    iset = st.session_state.get("intent_set")
    if not tools or iset is None:
        st.info("Load the tool list and an intent set first.")
    elif not api_key:
        st.warning("An Anthropic API key is required for the probes.")
    else:
        p1, p2, p3 = st.tabs(["Forced choice", "Pairwise discrimination", "Scaling curve"])

        with p1:
            st.caption(
                "Ranks tools per intent, several times, with stated reasons. Tells you the "
                "runner-up and the confidence, which a pass rate cannot."
            )
            c1, c2 = st.columns(2)
            samples = c1.number_input("Samples per intent", 1, 10, 5)
            temp = c2.slider("Temperature", 0.0, 1.0, 1.0, 0.1,
                             help="Above 0 so repeated picks reveal how stable the choice is.")
            shuffle = st.checkbox("Shuffle tool order between samples", True,
                                  help="If accuracy moves with ordering, descriptions are not carrying the decision.")
            if st.button("Run forced choice", type="primary"):
                bar = st.progress(0.0, text="starting")
                df = probes.sync(
                    probes.forced_choice(iset, tools, model, api_key, int(samples), temp,
                                         shuffle, lambda p, m: bar.progress(min(p, 1.0), text=m))
                )
                bar.empty()
                st.session_state["probe_fc"] = df

            if "probe_fc" in st.session_state:
                df = st.session_state["probe_fc"]
                c = st.columns(5)
                c[0].metric("Top-1 accuracy", f"{df['top1_acc'].mean():.1f}%")
                c[1].metric("Top-3 accuracy", f"{df['top3_acc'].mean():.1f}%")
                c[2].metric("Mean confidence", f"{df['mean_confidence'].mean():.2f}")
                c[3].metric("Confidently wrong", f"{100 * df['confidently_wrong'].mean():.0f}%")
                c[4].metric("Mean entropy", f"{df['entropy'].mean():.2f}")

                st.markdown("**Diagnosis**")
                near = df[(df["top1_acc"] < 50) & (df["top3_acc"] >= 80)]
                gone = df[df["top3_acc"] < 40]
                cw = df[df["confidently_wrong"]]
                amb = df[df["entropy"] > 0.6]
                notes = []
                if len(near):
                    notes.append(f"**{len(near)} intents** pick wrong but have the right tool in the top 3 — "
                                 "one sentence in the description fixes these.")
                if len(gone):
                    notes.append(f"**{len(gone)} intents** never surface the right tool at all — "
                                 "those tools are invisible; rename or remove them.")
                if len(cw):
                    notes.append(f"**{len(cw)} intents** are confidently wrong — the description actively misleads.")
                if len(amb):
                    notes.append(f"**{len(amb)} intents** have high entropy — the surface is genuinely ambiguous there.")
                st.markdown("\n\n".join(notes) if notes else "No threshold-crossing patterns.")

                st.dataframe(
                    df[["intent_id", "stage", "expected", "top_pick", "runner_up", "top1_acc",
                        "top3_acc", "mean_rank_of_correct", "mean_confidence",
                        "confidently_wrong", "entropy", "pick_spread"]],
                    use_container_width=True, hide_index=True,
                )

                st.markdown("**Stated reasons** — read these. They name the phrase that misled the model.")
                for _, r in df[df["top1_acc"] < 100].head(20).iterrows():
                    st.markdown(f"`{r['intent_id']}` expected **{r['expected']}**, picked **{r['top_pick']}**  \n"
                                f"_{r['all_reasons']}_")

                st.download_button("Export probe CSV", df.to_csv(index=False).encode(),
                                   file_name="forced_choice.csv")

        with p2:
            st.caption(
                "Presents only two tools. If the model cannot separate them head to head, "
                "it will never separate them among thirty."
            )
            names = sorted(tool_names)
            c1, c2 = st.columns(2)
            ta = c1.selectbox("Tool A", names, key="pa")
            tb = c2.selectbox("Tool B", names, index=min(1, len(names) - 1), key="pb")
            if st.button("Run pairwise", disabled=ta == tb):
                by = {t.name: t for t in tools}
                bar = st.progress(0.0, text="starting")
                res = probes.sync(
                    probes.pairwise_discrimination(by[ta], by[tb], iset.intents, model, api_key,
                                                   lambda p, m: bar.progress(min(p, 1.0), text=m))
                )
                bar.empty()
                if res["n"] == 0:
                    st.warning("No intents in the set reference either tool.")
                else:
                    st.metric(res["pair"], f"{res['accuracy']}%", help=f"{res['n']} intents")
                    st.info(res["verdict"])
                    st.dataframe(res["detail"], use_container_width=True, hide_index=True)

        with p3:
            st.caption(
                "Same questions, more distractors. Every list contains the correct tool, so the "
                "only variable is how many other options the model is weighing."
            )
            n_tools = len(tools)
            default = [s for s in [5, 10, 20, n_tools] if s <= n_tools]
            sizes = st.multiselect("List sizes", sorted(set(range(3, n_tools + 1))), default)
            if st.button("Run scaling curve", disabled=not sizes):
                bar = st.progress(0.0, text="starting")
                curve = probes.sync(
                    probes.scaling_curve(iset, tools, model, api_key, sizes, 0,
                                         lambda p, m: bar.progress(min(p, 1.0), text=m))
                )
                bar.empty()
                st.session_state["probe_curve"] = curve
            if "probe_curve" in st.session_state:
                curve = st.session_state["probe_curve"]
                st.line_chart(curve.set_index("tools_in_list")["top1_accuracy"])
                st.dataframe(curve, use_container_width=True, hide_index=True)
                if len(curve) > 1:
                    drop = curve.iloc[0]["top1_accuracy"] - curve.iloc[-1]["top1_accuracy"]
                    if drop > 5:
                        st.error(
                            f"Accuracy falls {drop:.1f} points between "
                            f"{int(curve.iloc[0]['tools_in_list'])} and "
                            f"{int(curve.iloc[-1]['tools_in_list'])} tools on identical questions. "
                            "The tool count itself is costing you."
                        )
                    else:
                        st.success("Accuracy is flat across list sizes. The count is not the bottleneck.")


# ---------------------------------------------------------------- report

with tab_report:
    st.header("Evaluation report")
    runs = store.list_runs(conn)
    if not runs:
        st.info("No runs stored yet.")
    else:
        rid = st.selectbox(
            "Run", [r["run_id"] for r in runs],
            format_func=lambda x: f"{x} — {dict([r for r in runs if r['run_id'] == x][0])['label']}",
        )
        meta = dict([r for r in runs if r["run_id"] == rid][0])
        df = metrics.intent_frame(conn, rid)
        calls = metrics.call_frame(conn, rid)

        if df.empty:
            st.warning("This run has no results.")
        else:
            enabled = json.loads(meta["tools_enabled"] or "[]")
            st.caption(
                f"**{meta['label']}** · {meta['runner']} · model `{meta['model'] or 'none'}` · "
                f"tools `{meta['tools_hash']}` · {meta['repeats']} repeat(s) · {meta['created_at']}"
            )

            st.subheader("Headline")
            head = metrics.headline(df)
            cols = st.columns(5)
            order = [
                "Resolution rate", "First-call accuracy", "Tool selection accuracy",
                "Stop-short rate", "Zero-call rate",
            ]
            for col, key in zip(cols, order):
                col.metric(key, f"{head[key]:.1f}%")
            cols2 = st.columns(5)
            cols2[0].metric("Fishing rate", f"{head['Fishing rate']:.1f}%")
            cols2[1].metric("Mean calls / intent", f"{head['Mean calls per intent']:.2f}")
            cols2[2].metric("Median tool tokens", f"{head['Median tool tokens per intent']:.0f}")
            cols2[3].metric("p95 tool tokens", f"{head['p95 tool tokens per intent']:.0f}")
            cols2[4].metric("Median latency", f"{head['Median latency (ms)']:.0f} ms")

            st.subheader("What to change, ranked")
            util = metrics.tool_utilisation(calls, enabled, df["intent_id"].nunique())
            stages = metrics.by_stage(df)
            recs = metrics.recommendations(df, util, stages, st.session_state.get("audit"))
            if recs:
                st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
            else:
                st.success("No threshold-crossing problems detected in this run.")

            st.subheader("Failure taxonomy")
            tags = metrics.tag_distribution(df)
            c1, c2 = st.columns([2, 3])
            c1.dataframe(tags, use_container_width=True, hide_index=True)
            c2.dataframe(
                pd.DataFrame(
                    [{"tag": k, "meaning": v[0], "fix": v[1]} for k, v in TAG_MEANING.items()]
                ),
                use_container_width=True, hide_index=True,
            )

            st.subheader("Tool utilisation")
            st.caption("Dead tools and token hogs. Confirm with ablation before removing anything.")
            st.dataframe(util, use_container_width=True, hide_index=True)

            st.subheader("Routing confusion")
            st.caption("Rows: the tool that should have been called first. Columns: what was called.")
            cm = metrics.confusion(df)
            if not cm.empty:
                st.dataframe(cm.style.background_gradient(cmap="Reds"), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.subheader("By workflow stage")
            c1.caption("Weak stages have no effective entry point.")
            c1.dataframe(stages, use_container_width=True, hide_index=True)
            c2.subheader("By question shape")
            c2.caption("Adopter-phrased versus agent-harvested.")
            c2.dataframe(metrics.by_shape(df), use_container_width=True, hide_index=True)

            stab = metrics.stability(df)
            if not stab.empty:
                st.subheader("Stability across repeats")
                st.caption("Intents that flip between runs usually sit on a tool boundary the model cannot see.")
                st.dataframe(stab.head(25), use_container_width=True, hide_index=True)

            abl = st.session_state.get("ablation")
            if abl and abl["baseline"] == rid:
                st.subheader("Leave-one-out ablation")
                st.caption("Negative marginal value means the tool was making the agent worse.")
                st.dataframe(
                    metrics.ablation_table(conn, rid, abl["runs"]),
                    use_container_width=True, hide_index=True,
                )

            st.subheader("Trace inspector")
            st.caption("Attach three of these to every claim you make. Aggregates persuade nobody on their own.")
            only_failed = st.checkbox("Failures only", True)
            view = df[df["resolved"] == 0] if only_failed else df
            pick = st.selectbox("Intent", view["intent_id"].unique() if not view.empty else [])
            if pick:
                row = view[view["intent_id"] == pick].iloc[0]
                st.write(
                    f"**tag** `{row['tag']}` · expected {row['expected_tools']} · "
                    f"called {row['called_tools']} · {row['n_calls']} calls · "
                    f"{row['tokens_tools']} tool tokens · {row['latency_ms']} ms"
                )
                trace = calls[calls["intent_id"] == pick].sort_values(["repeat_idx", "call_idx"])
                for _, c in trace.iterrows():
                    with st.expander(
                        f"#{c['call_idx']} {c['tool']} — {c['resp_tokens']} tok, "
                        f"{c['latency_ms']} ms, hit={bool(c['hit'])}"
                    ):
                        st.code(c["args"], language="json")
                        st.text(c["resp_excerpt"] or c["error"])
                st.markdown("**Final answer**")
                st.info(row["final_answer"] or "(empty)")

            st.download_button(
                "Export report data (CSV)",
                df.drop(columns=["final_answer"]).to_csv(index=False).encode(),
                file_name=f"{rid}_results.csv",
            )

# ---------------------------------------------------------------- compare

with tab_compare:
    st.header("Before and after")
    runs = store.list_runs(conn)
    if len(runs) < 2:
        st.info("Two runs are needed to compare.")
    else:
        ids = [r["run_id"] for r in runs]
        c1, c2 = st.columns(2)
        a = c1.selectbox("Baseline", ids, index=min(1, len(ids) - 1))
        b = c2.selectbox("Candidate", ids, index=0)
        da, db = metrics.intent_frame(conn, a), metrics.intent_frame(conn, b)

        ma = dict([r for r in runs if r["run_id"] == a][0])
        mb = dict([r for r in runs if r["run_id"] == b][0])
        if ma["intent_set"] != mb["intent_set"]:
            st.error(
                f"Different intent sets (`{ma['intent_set']}` vs `{mb['intent_set']}`). "
                "The comparison is not valid."
            )
        if ma["model"] != mb["model"]:
            st.warning(
                f"Different models (`{ma['model']}` vs `{mb['model']}`). "
                "You are measuring model drift as well as your change."
            )

        if not da.empty and not db.empty:
            table = metrics.compare_runs(da, db, ma["label"] or a, mb["label"] or b)
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.subheader("Intents that changed")
            ra = da.groupby("intent_id")["resolved"].mean()
            rb = db.groupby("intent_id")["resolved"].mean()
            joined = pd.DataFrame({"baseline": ra, "candidate": rb}).dropna()
            joined["delta"] = joined["candidate"] - joined["baseline"]
            moved = joined[joined["delta"] != 0].sort_values("delta")
            st.caption(f"{(moved['delta'] > 0).sum()} fixed, {(moved['delta'] < 0).sum()} regressed.")
            st.dataframe(moved.reset_index(), use_container_width=True, hide_index=True)
