# VDS MCP Eval

A harness for measuring how an MCP server actually behaves under an agent, and
for proving that a change to it made things better.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Connect to your server in the sidebar (http, sse or stdio), pick an intent set,
and run.

---

## Why it is shaped this way

An MCP server can fail in three unrelated places, and each has a different fix:

| Failure | Detected by | Fix |
|---|---|---|
| Wrong tool called | selection accuracy, confusion matrix | tool names and descriptions |
| Right tool, nothing useful back | hit rate | search index, synonyms, data |
| Right content, agent did not use it | stop-short rate, payload size | response shape and size |

If you only measure "did the agent answer correctly", you cannot tell these
apart, and you will rewrite documentation when the real problem was a sentence
in a tool description.

Everything is stored as raw traces in SQLite. Metrics are recomputed from traces
on every view, so a run captured today can be re-scored against a rule you
invent in three months without re-running it.

---

## The four runners

| Runner | Model? | Cost | Isolates |
|---|---|---|---|
| **R0 static** | no | free | The tool surface itself. Description overlap, schema quality, payload size. |
| **R1 direct** | no | free | Retrieval quality. Calls each intent's expected tools directly. |
| **R2 routing** | yes | low | Tool selection. One turn, one intent, N repeats. |
| **R3 task** | yes | higher | The whole loop on multi-turn build tasks. |
| **Selection probe** | yes | low | How the model *reads* the tool list. No MCP connection at all. |

**The most useful comparison is R1 against R2.** If R1 resolves an intent and R2
does not, the data was there and the agent could not find it. That is a
description problem, and it is cheap to fix.

Start with R0. It needs no intent set, no API key and no model budget, and it
produces publishable evidence in about a minute.

---

## The intent set

`intents/baseline-v1.json` ships with 20 worked examples. Extend it to ~100:
roughly 60 adopter-shaped and 40 agent-shaped.

**Adopter-shaped** intents are written by hand from real sources: your support
channel, zero-result docsite searches, GitHub issues, and PR review comments
where someone wrote "use X instead" (those are pure selection failures and they
are the best items in the set). Keep the original phrasing, vagueness included.

**Agent-shaped** intents are *harvested*, not written. Agents do not ask what
humans ask — they emit terse fragments derived from their own guess, usually
after they have already picked a component. Run 20 real build tasks with your
MCP connected, log every query the agent emits, and that log is your agent set.

Each intent carries:

| Field | Why it exists |
|---|---|
| `expected_tools` / `acceptable_tools` | scores routing |
| `answer_contains` | scores correctness automatically |
| `stage` | produces coverage-by-workflow-stage, the positioning answer |
| `shape` | splits adopter phrasing from agent phrasing |
| `answerable` | counts coverage gaps without running anything |

### Freeze it

Before the baseline run, commit the intent file and do not edit it again. Append
a `baseline-v2.json` later if you must, but never change v1 in place — the
moment the questions move, your before-and-after stops being an argument.

The same applies to the model string. The app records the model, the server
fingerprint and the enabled tool list on every run, and the Compare tab warns
you when two runs are not comparable.

---

## What the report tells you

**Headline** — resolution rate, first-call accuracy, tool selection accuracy,
stop-short rate, zero-call rate, fishing rate, calls per intent, token cost,
latency.

**What to change, ranked** — threshold-crossing findings turned into actions,
each naming the evidence behind it.

**Failure taxonomy** — every intent gets exactly one tag:

| Tag | Meaning | Fix |
|---|---|---|
| `R` | wrong or no tool called | rewrite descriptions, declare boundaries |
| `C` | no tool can answer this | add a tool |
| `Q` | right tool, no match | search index, synonyms |
| `D` | right match, thin content | documentation backlog |
| `P` | right content, agent did not use it | shrink or reshape the payload |

The tag distribution is your work plan. 60% `R` means the fix is descriptions and
you should not touch a single doc page. 60% `D` means your server is fine and the
content is thin.

**Tool utilisation** — calls, share, hit rate, token cost and latency per tool,
including tools that were never called at all.

**Routing confusion** — expected tool against tool actually called first. This is
the artifact that shows you *which pairs* are ambiguous rather than just that
something is wrong.

**By workflow stage** — discover, decide, specify, compose, validate, meta. A
stage with low resolution has no effective entry point. This is how you answer
"are the tools positioned correctly."

**Stability** — intents that flip between repeats. Unstable routing almost always
means two tools the model cannot separate.

**Trace inspector** — the full call sequence for any intent, with arguments,
response excerpts, tokens and latency.

---

## The selection probe

R2 records *which* tool the model chose. The probe records *how it chose* — and
that distinction decides your fix.

At selection time the model never sees your data. It sees names, descriptions and
schemas. Your whole server is, at that moment, about thirty paragraphs of text.
So tool organisation can be measured with no MCP connection and no tool
execution.

The probe hands the model the tool list and one question, and requires a ranked
answer with reasons and confidence. Four things fall out that a pass rate cannot
give you:

| Signal | What it means |
|---|---|
| **Rank of the correct tool** | Second place is a one-sentence fix. Absent from the top five means the tool is invisible. Same failure, opposite fix. |
| **Stated reason** | Read fifty and you will find the exact phrase that misleads. "Unclear how this differs from X" is the model telling you to merge. |
| **Confidence** | A correct pick at 0.35 confidence is a coin flip that landed well and will flip on another model. |
| **Entropy across samples** | Low entropy and wrong means the description actively misleads. High entropy means the surface is genuinely ambiguous. |

The dangerous quadrant is confidently wrong, and it is invisible in an aggregate
score.

### Pairwise discrimination

Present only two tools and ask questions that should route to each. Roughly 50%
on a two-way choice is a coin flip: the tools are indistinguishable, and if the
model cannot separate them head to head it will never separate them among thirty.
This is the cleanest merge argument available.

### Scaling curve

The same questions asked with 5, 10, 20 and then all tools in the list, where
every list contains the correct tool plus random distractors. The only variable
is how many other options the model is weighing. A downward slope means the count
itself is costing you accuracy — and you have a chart that says so rather than an
opinion.

---

## Answering "too many tools or too few"

**Too many** — run leave-one-out ablation from the Run tab. Each selected tool is
disabled and the whole set re-run. The drop in resolution is that tool's marginal
value:

- zero → merge or delete it
- negative → it was actively confusing the model
- above a point or two → it is earning its place

**Too few** — read the coverage numbers. The `answerable` count gives you the
gap before you spend anything, tag `C` gives it to you after, and the fishing
rate (same tool called three or more times with rephrasings) tells you where an
entry point is missing.

Ablation is expensive: one full re-run per tool. Start with the three or four
tools the R0 similarity table already flags as duplicates.

---

## Suggested sequence

1. **R0** on the live server. Publish the overlap table and payload profile.
2. Write 40 adopter intents from support history. Run **R1**. Freeze the file.
3. Harvest 40 agent intents from 20 real task runs. Run **R2** with 3 repeats.
4. Ablate the pairs R0 flagged.
5. Publish the baseline report, dated, with three real traces attached to every
   claim. Aggregates persuade nobody on their own — a stakeholder who watches the
   agent call the wrong tool and answer confidently will believe the number.

Then change the server, re-run, and open the Compare tab.

---

## Layout

```
app.py                    Streamlit UI: audit, run, report, compare
eval_core/
  mcp_client.py           transports, tool listing, tool calls, fingerprinting
  store.py                SQLite trace store
  intents.py              intent loading, validation, failure tagging
  runners.py              R0/R1/R2/R3 and the agent loop
  metrics.py              all aggregation, ablation, recommendations
intents/baseline-v1.json  seed intent set
runs/traces.db            created on first run
```

## Notes

- Token counts for tool responses are estimated at four characters per token.
  Absolute values are approximate; the comparison between runs is what matters,
  and the estimate is consistent.
- `answer_contains` drives automatic scoring. Intents left with an empty list are
  scored on tool selection alone and should be reviewed by hand.
- The agent loop caps at `max_turns`. Intents that hit the cap are recorded as
  unresolved, which is the correct outcome.
