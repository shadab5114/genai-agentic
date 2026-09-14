"""Intent set loading, validation, and failure tagging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INTENT_DIR = Path(__file__).resolve().parent.parent / "intents"

STAGES = ["discover", "decide", "specify", "compose", "validate", "meta"]
SHAPES = ["adopter", "agent"]

TAG_MEANING = {
    "OK": ("Resolved correctly", "nothing to do"),
    "R": ("Routing failure - wrong or no tool called", "rewrite tool names and descriptions"),
    "C": ("Coverage failure - no tool can answer this", "add or extend a tool"),
    "Q": ("Query failure - right tool, no useful match", "fix search index, synonyms, aliases"),
    "D": ("Data failure - right match, content insufficient", "fix the underlying documentation"),
    "P": ("Payload failure - right content, agent did not use it", "shrink or reshape the response"),
}


@dataclass
class Intent:
    id: str
    text: str
    expected_tools: list[str] = field(default_factory=list)
    acceptable_tools: list[str] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)
    shape: str = "adopter"
    stage: str = "discover"
    source: str = "manual"
    answerable: bool = True
    notes: str = ""

    @property
    def ok_tools(self) -> set[str]:
        return set(self.expected_tools) | set(self.acceptable_tools)


@dataclass
class IntentSet:
    version: str
    intents: list[Intent]
    path: Path | None = None

    def by_id(self, intent_id: str) -> Intent | None:
        return next((i for i in self.intents if i.id == intent_id), None)


def load_intent_set(path: str | Path) -> IntentSet:
    path = Path(path)
    raw = json.loads(path.read_text())
    intents = [Intent(**item) for item in raw["intents"]]
    return IntentSet(version=raw.get("version", path.stem), intents=intents, path=path)


def available_sets() -> list[Path]:
    INTENT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(INTENT_DIR.glob("*.json"))


def validate(intent_set: IntentSet, tool_names: set[str]) -> list[str]:
    """Return human-readable problems with the intent set."""
    problems: list[str] = []
    seen: set[str] = set()
    for i in intent_set.intents:
        if i.id in seen:
            problems.append(f"{i.id}: duplicate id")
        seen.add(i.id)
        if i.stage not in STAGES:
            problems.append(f"{i.id}: unknown stage '{i.stage}'")
        if i.shape not in SHAPES:
            problems.append(f"{i.id}: unknown shape '{i.shape}'")
        unknown = i.ok_tools - tool_names
        if tool_names and unknown:
            problems.append(f"{i.id}: references tools not on the server: {sorted(unknown)}")
        if i.answerable and not i.expected_tools:
            problems.append(f"{i.id}: marked answerable but has no expected_tools")
    return problems


def _contains_any(haystack: str, needles: list[str]) -> bool:
    low = (haystack or "").lower()
    return any(n.lower() in low for n in needles) if needles else False


def tag_result(
    intent: Intent,
    called: list[str],
    tool_outputs: list[str],
    final_answer: str,
) -> tuple[str, bool]:
    """Assign one failure tag and decide whether the intent resolved.

    The ladder is deliberately ordered: each tag points at exactly one fix, so
    the tag distribution in the report doubles as a work plan.
    """
    answer_hit = _contains_any(final_answer, intent.answer_contains)
    if not intent.answer_contains:
        # No automatic ground truth -- treat "a correct tool was called" as resolved
        # and flag for manual review via the OK/R split only.
        answer_hit = bool(set(called) & intent.ok_tools)

    if answer_hit:
        return "OK", True

    if not intent.answerable:
        return "C", False

    if not called:
        return "R", False

    if not (set(called) & intent.ok_tools):
        return "R", False

    tool_hit = any(_contains_any(out, intent.answer_contains) for out in tool_outputs)
    if tool_hit:
        # The tool gave the agent the answer and the agent still did not use it.
        return "P", False

    non_empty = [o for o in tool_outputs if o.strip()]
    if not non_empty:
        return "Q", False

    return "D", False
