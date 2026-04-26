"""
Agent State — the data structure that flows through the LangGraph graph.

Every node in the graph receives the current state and returns an update.
LangGraph merges updates automatically.

Key concept — Annotated with operator.add:
  messages: Annotated[list, operator.add]
  This means: when a node returns new messages, APPEND them to existing ones.
  Without this, each node would overwrite the previous messages entirely.
  operator.add is how LangGraph knows to accumulate conversation history.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    The complete state of the agent at any point in the graph.

    This flows between every node — each node reads what it needs
    and returns only the fields it updates.
    """

    # Full conversation history — HumanMessage, AIMessage, ToolMessage
    # operator.add means new messages are APPENDED, not replaced
    messages: Annotated[list[BaseMessage], operator.add]

    # Which company the user is asking about
    # Set at the start, read by tools to scope their searches
    company_name: str

    # The final answer to return to the user
    # Set by the last node before END
    final_answer: str
