"""
LangGraph Agent Graph — the autonomous decision-making core.

This is the most important file in the project. It defines:
  - HOW the agent reasons (ReAct pattern)
  - WHEN it uses tools vs. generates a final answer
  - HOW conversation state flows between steps

Graph structure:
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌────▶│  agent node │ ◀──────────────┐
              │     │  (LLM call) │                │
              │     └──────┬──────┘                │
              │            │                       │
              │     ┌──────▼──────┐                │
              │     │  should we  │                │
              │     │  use tools? │                │
              │     └──────┬──────┘                │
              │            │                       │
              │      Yes ──┤── No                  │
              │            │    │                  │
              │     ┌──────▼──┐ │                  │
              │     │  tools  │ │                  │
              └─────│  node   │─┘                  │
                    │(execute)│────────────────────┘
                    └─────────┘         (loop back to agent)

The agent node calls the LLM.
If the LLM decides to use a tool → tools node executes it → loops back.
If the LLM has enough info → goes to END with final answer.

This loop is what makes it "autonomous" — it decides on its own
how many tool calls to make before it's confident enough to answer.

Interview talking point:
  "I built the agent with LangGraph's StateGraph using a ReAct pattern.
  The conditional edge decides whether to execute tools or generate the
  final answer — this decision loop runs automatically until the LLM
  judges it has sufficient context."
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.agent.prompts import AGENT_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config import settings


def _build_agent_node():
    """
    Build the agent node function.

    The agent node is the LLM with tools bound to it.
    Binding tools tells the LLM: "you CAN call these functions."
    The LLM decides WHICH tool to call (or none) based on the conversation.
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    # bind_tools() adds the tool schemas to every LLM call
    # The LLM sees tool names + descriptions and can choose to call them
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """
        The reasoning node — calls the LLM to decide next action.

        Receives: full conversation history (messages)
        Returns: the LLM's response (either a tool call or a final answer)
        """
        logger.info(f"Agent node | messages={len(state['messages'])}")

        # Prepend system prompt if this is the first call
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

        # Call the LLM — it returns either:
        # - An AIMessage with tool_calls → means "call these tools"
        # - An AIMessage with just content → means "here's my final answer"
        response = llm_with_tools.invoke(messages)
        logger.info(
            f"Agent response | "
            f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') else 0} | "
            f"content_length={len(response.content)}"
        )

        # Return dict — LangGraph merges this into state
        # messages uses operator.add so this APPENDS, not replaces
        return {"messages": [response]}

    return agent_node


def _should_continue(state: AgentState) -> str:
    """
    Conditional edge — decides whether to use tools or end.

    This function is called after every agent node execution.
    It reads the last message and checks if the LLM requested tool calls.

    Returns:
        "tools"  → go to tools node (LLM wants to call a tool)
        "end"    → go to END (LLM has a final answer)
    """
    last_message = state["messages"][-1]

    # If the LLM's response contains tool_calls, it wants to use a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"Routing to tools: {tool_names}")
        return "tools"

    # No tool calls — the LLM has generated a final answer
    logger.info("Routing to END — final answer ready")
    return "end"


def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent graph.

    Returns a compiled graph that can be invoked like:
        result = graph.invoke({"messages": [...], "company_name": "Danone"})
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    agent_node = _build_agent_node()
    graph.add_node("agent", agent_node)

    # ToolNode is LangGraph's built-in tool executor
    # It reads tool_calls from the last message, runs each tool,
    # and returns ToolMessages with the results
    tool_node = ToolNode(tools=AGENT_TOOLS)
    graph.add_node("tools", tool_node)

    # --- Set entry point ---
    graph.set_entry_point("agent")

    # --- Add edges ---

    # After agent node: decide whether to use tools or finish
    graph.add_conditional_edges(
        "agent",              # From this node
        _should_continue,     # Call this function to decide
        {
            "tools": "tools", # If returns "tools" → go to tools node
            "end": END,       # If returns "end" → finish
        },
    )

    # After tools node: ALWAYS go back to agent
    # (agent will see tool results and decide what to do next)
    graph.add_edge("tools", "agent")

    # Compile into a runnable
    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")
    return compiled


# ─────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────

class CompanyResearchAgent:
    """
    High-level wrapper around the LangGraph agent.

    Usage:
        agent = CompanyResearchAgent()
        result = agent.research("What is Danone's revenue?", company="Danone")
        print(result["answer"])
    """

    def __init__(self):
        self.graph = build_agent_graph()
        logger.info("CompanyResearchAgent ready")

    def research(self, question: str, company_name: str) -> dict:
        """
        Answer a question about a company using autonomous tool selection.

        Args:
            question: Natural language question from the user
            company_name: Company to research

        Returns:
            dict with keys: answer (str), messages (list), tool_calls_made (int)
        """
        logger.info(f"Research request | company={company_name} | q={question}")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "company_name": company_name,
            "final_answer": "",
        }

        # Run the graph — it loops until the agent decides to stop
        result = self.graph.invoke(initial_state)

        # Extract the final answer from the last AIMessage
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else ""

        # Count how many tool calls were made
        tool_calls_made = sum(
            1 for m in result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        )

        logger.info(f"Research complete | tool_calls={tool_calls_made} | answer_length={len(answer)}")

        return {
            "answer": answer,
            "messages": result["messages"],
            "tool_calls_made": tool_calls_made,
            "company_name": company_name,
        }
