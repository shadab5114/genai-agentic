# Cheat Sheet: LangChain Agents

## I. Understanding built-in agents

LangChain provides built-in agents that enable LLMs to make decisions, use tools, access memory, and interact with external systems. These agents are typically accessed in two ways:

- **Prebuilt agent creators** — Utility functions that help quickly set up ready-to-use agents for common use cases (for example, SQL, CSV, Pandas).
- **LangGraph ReAct agents** — Graph-structured agents with explicit reasoning steps and control flow logic.

> **Version note:** These examples were verified against `langchain` 1.3.16 / `langchain-classic` 1.0.8 / `langgraph` 1.2.11. In this generation of LangChain, the legacy agent APIs below (`initialize_agent`, `AgentType`, `create_tool_calling_agent`, etc.) live in **`langchain_classic.agents`**, not `langchain.agents` — that top-level module now points at the newer `create_agent` API instead (see the end of this document).

## II. Exploring core agent types

These agent types define the basic way an agent thinks, chooses tools, and performs actions. They are foundational and can be used to build more specialized agents.

| Agent type | Description |
|---|---|
| `ZERO_SHOT_REACT_DESCRIPTION` | Performs reasoning before acting. Picks the right tool using only its description. |
| `REACT_DOCSTORE` | Zero-shot agent with access to a document store for retrieving info (for example, Wikipedia). |
| `SELF_ASK_WITH_SEARCH` | Breaks complex questions into simpler ones and answers using a search tool. |
| `CONVERSATIONAL_REACT_DESCRIPTION` | Maintains conversation history while reasoning and acting. |
| `CHAT_ZERO_SHOT_REACT_DESCRIPTION` | Like a zero-shot agent but optimized for chat models like GPT-4. |
| `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` | Chat-based version of conversational ReAct agent. |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` | Optimized for chat models and structured tools with multiple inputs. |
| `OPENAI_FUNCTIONS` | Designed to work with OpenAI's function calling schema. |
| `OPENAI_MULTI_FUNCTIONS` | Handles multiple tools using OpenAI's multi-function architecture. |

## III. Model compatibility

Some LLMs may not fully support structured output parsing required by certain agents (for example, `structured-chat-zero-shot-react-description`). If the tool returns a dictionary or complex output, these models might fail with parsing or validation errors.

**Description:** You can create an agent using `initialize_agent`, where you pass the agent type, so it behaves according to the agent type. The agent in this code will perform reasoning before acting.

**Code:**

```python
from langchain_classic.agents import initialize_agent, AgentType
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

agent = initialize_agent(
    tools=[get_word_length],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.invoke({"input": "How many letters are in the word educative?"})
print(result["output"])
```

Running this prints the model's reasoning trace before its answer:

```
> Entering new AgentExecutor chain...
I need to find the number of characters in the word "educative".
Action: get_word_length
Action Input: "educative"
Observation: 9
Thought: I now know the final answer
Final Answer: 9
> Finished chain.
9
```

> **Note on tool arity:** `ZERO_SHOT_REACT_DESCRIPTION` (and other legacy single-input ReAct agents) can only use **single-input** tools — they parse the model's output as one raw string per action. A multi-argument tool like `multiply(a: int, b: int)` raises `ValueError: ZeroShotAgent does not support multi-input tool multiply` at agent-creation time. Use `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` (or a native tool-calling agent — see section V) for tools with multiple parameters.

## IV. Prebuilt agents (task-specific utilities)

These utilities use the core agent types behind the scenes but make it easy to set up agents for specific types of data or tasks.

| Function | Purpose | Agent type (internally used by default) |
|---|---|---|
| `create_pandas_dataframe_agent()` | Natural language to DataFrame analysis and visualization | `zero-shot-react-description` |
| `create_csv_agent()` | Query CSV files conversationally | `zero-shot-react-description` |
| `create_sql_agent()` | Natural language SQL queries | `zero-shot-react-description` |
| `create_openai_functions_agent()` | Agent with OpenAI function-calling tools | `openai-functions` |
| `create_tool_calling_agent()` | Generic agent that invokes structured tools | `structured-chat-zero-shot-react-description` |

> **Package note:** `create_sql_agent()` ships in `langchain_community.agent_toolkits`. `create_pandas_dataframe_agent()` and `create_csv_agent()` were moved out to the separate **`langchain_experimental`** package, which is community-maintained and not installed by default — install it explicitly (`pip install langchain_experimental pandas`) if you need these two.

**Description:** This code shows an example of `create_pandas_dataframe_agent` where you pass in the LLM, the dataframe you are working on, the agent type, and `verbose`. You can use this to perform dataframe analysis using natural language.

**Code:**

```python
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import init_chat_model

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "age": [30, 25, 35],
})

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="zero-shot-react-description",
    verbose=True,
    allow_dangerous_code=True,  # required: the agent executes generated Python against your dataframe
)

result = agent.invoke({"input": "Who is the oldest person in the dataframe?"})
print(result["output"])
# -> "The oldest person in the dataframe is Carol, who is 35 years old."
```

`allow_dangerous_code=True` is required because this agent works by having the LLM write and execute real Python (`python_repl_ast`) against your dataframe — treat it the same as any other arbitrary code execution surface, and don't point it at untrusted data or run it with elevated privileges.

This is also a good illustration of the section III warning in practice: in testing, `gpt-4o-mini` occasionally emitted malformed `Action:` lines the legacy ReAct parser didn't recognize (`"... is not a valid tool, try one of [python_repl_ast]"`), retried several times, and ultimately reasoned its way to the correct answer in text rather than by successfully invoking the tool. The final answer was still correct, but the interaction is noticeably less reliable than the native tool-calling agents in section V.

## V. LangGraph agents

LangGraph supports building multi-step reasoning agents using a directed graph structure. These agents are ideal for advanced use cases where you want full control over reasoning steps, looping, or conditional logic.

**Description:** The `create_react_agent()` utility constructs a graph agent that performs tool usage in a loop with reasoning at each step until a stopping condition is met.

**Code:**

```python
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
graph_agent = create_react_agent(llm, tools=[multiply])

result = graph_agent.invoke({"messages": [("user", "What is 7 times 6?")]})
print(result["messages"][-1].content)
# -> "7 times 6 is 42."
```

Unlike `ZERO_SHOT_REACT_DESCRIPTION`, this handles the multi-argument `multiply` tool without issue — it relies on the model's native tool-calling support rather than parsing a single text action string.

> **Deprecation note:** `langgraph.prebuilt.create_react_agent` is deprecated as of LangGraph v1.0 (still functional, removal planned for v2.0) in favor of `langchain.agents.create_agent`, which is a drop-in replacement in this version:
>
> ```python
> from langchain.agents import create_agent
>
> agent = create_agent(llm, tools=[multiply])
> result = agent.invoke({"messages": [("user", "What is 7 times 6?")]})
> ```
>
> For new work, prefer `langchain.agents.create_agent` over both `langgraph.prebuilt.create_react_agent` and the legacy `langchain_classic.agents.initialize_agent` family described above — it's the actively maintained entry point in this LangChain generation, built on the same underlying loop as the recursive tool-calling pattern in [tool_calling_agent](../tool_calling_agent/README.md).

## See also

- [manual-tool-calling-cheat-sheet.md](manual-tool-calling-cheat-sheet.md) — how tool calls, `ToolMessage`, and `tool_call_id` work under the hood, which every agent type in this document relies on.
- [tool_calling_agent/README.md](../tool_calling_agent/README.md) — a hand-built version of the same "reason, call a tool, observe, repeat" loop these prebuilt agents automate.
