


from typing import Literal

from langgraph.types import Command, Send

from pydantic import BaseModel, Field, Field

from config import llm
from state import StackState
from langchain_core.messages import SystemMessage, HumanMessage

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator for SnackStack, a voice-enabled food delivery assistant.

YOUR JOB:
Analyze the customer's query (and conversation history if provided) and decide
which specialist agents to dispatch.

AGENTS AVAILABLE:
- menu_agent  → menu discovery, dish recommendations, dietary queries, general conversation
- order_agent → order tracking, order status inquiries

RULES:
- If the query is a greeting or general chat (hi, hello, hey, thanks, etc.) → route to [menu_agent]
- If the query is about food/menu → route to [menu_agent]
- If the query is ONLY about an order → route to [order_agent]
- If the query spans BOTH topics → route to [menu_agent, order_agent]
- Use conversation history to understand vague follow-ups. E.g. if the customer
  previously asked about non-veg and now says "anything works", that's still a
  menu query — route to [menu_agent].
- When in doubt, default to [menu_agent]
"""


class OrchestratorDecision(BaseModel):
    """The orchestrator's routing decision."""

    reasoning: str = Field(description="Brief explanation of why these agents were chosen")
    agents: list[Literal["menu_agent", "order_agent"]] = Field(
        description="List of agents to dispatch. Always at least one.",
        min_length=1,
    )

routing_llm = llm.with_structured_output(OrchestratorDecision)

def orchestrator_node(state:StackState) -> Command[Literal["menu_agent_node", "order_agent_node"]]:
    """Classify the query and fan out to one or both agents."""
    query = state["user_query"]

    history = state.get("messages", [])[-6:]

    decision: OrchestratorDecision = routing_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        *history,
        HumanMessage(content=query),
    ])

    clean_state = {
        **state,
        "menu_messages": [],
        "order_messages": [],
        "menu_response": "",
        "order_response": "",
    }

    sends = [Send(f"{agent}_node", clean_state) for agent in decision.agents]
    return Command(goto=sends, update={"route": decision.agents})

