

# Patterns to detect identifiers in the user query
import re
from config import llm
from state import StackState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool, ToolException
from langgraph.types import interrupt
from tools.order_tools import format_order, find_order
from tools.order_tools import order_tools_list

ORDER_ID_RE = re.compile(r"ORD-\d+", re.IGNORECASE)
TRACKING_RE = re.compile(r"SS\d+TRK", re.IGNORECASE)
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

order_llm = llm.bind_tools(order_tools_list)

ORDER_AGENT_PROMPT = """\
You are the Order Support Agent for SnackStack.

YOUR ROLE:
- Track orders and provide real-time status updates

AVAILABLE TOOLS:
1. get_order_status — look up an order by Order ID, Tracking ID, or email

GUIDELINES:
- The user's query will include a "Lookup key: ..." line — use that value
  when calling get_order_status.
- Be empathetic and professional.
- If the order is not found, ask the customer to double-check the identifier.
"""

def extract_identifier(text: str) -> str | None:
    """Try to extract an order ID, tracking ID, or email from text."""
    for pattern in (ORDER_ID_RE, TRACKING_RE, EMAIL_RE):
        match = pattern.search(text)
        if match:
            return match.group()
    return None

def order_agent_node(state: StackState) -> dict:
    """Call the LLM. On first invocation handle HITL for missing IDs;
    on subsequent invocations continue after tool results."""

    existing = state.get("order_messages") or []

    if not existing:
        # First call — extract identifier or interrupt
        # logger.info("Processing order query...")
        query = state["user_query"]
        lookup_key = extract_identifier(query)

        # if lookup_key:
        #     # logger.info("Found identifier: %s", lookup_key)
        # else:
            # logger.info("No identifier found — interrupting for user input")
        lookup_key = interrupt(
            "I'd be happy to help with your order! "
            "Could you please provide one of the following?\n"
            "  • Order ID    (e.g. ORD-201)\n"
            "  • Tracking ID (e.g. SS201TRK)\n"
            "  • Email       (e.g. priya@example.com)"
        )
        lookup_key = lookup_key.strip()

        all_msgs = [
            SystemMessage(content=ORDER_AGENT_PROMPT),
            HumanMessage(content=f"{query}\n\nLookup key: {lookup_key}"),
        ]
    else:
        # Subsequent call — tool results already in order_messages
        # ogger.info("Order agent re-invoked after tool results")
        all_msgs = existing

    response = order_llm.invoke(all_msgs)
    all_msgs = [*all_msgs, response]

    has_tools = bool(getattr(response, "tool_calls", None))
    # logger.info("Tool calls: %s", has_tools)

    update: dict = {"order_messages": all_msgs}

    if not has_tools:
        update["order_response"] = response.content
        update["messages"] = [response]

    return update
