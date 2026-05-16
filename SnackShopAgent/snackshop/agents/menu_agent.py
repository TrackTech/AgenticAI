from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from state import StackState
from config import llm
from tools.menu_tools import menu_tools_list

MENU_AGENT_PROMPT = """\
You are the Menu Discovery Agent for SnackShop, a food delivery platform.

YOUR ROLE:
- Help customers find dishes they'll love
- Provide detailed info about ingredients, dietary tags, and prices
- Handle general greetings and conversation warmly

AVAILABLE TOOLS:
1. search_menu_catalog — semantic search over our live menu (RAG)

GUIDELINES:
- For greetings (with no food context), respond warmly and offer to help.
- For ANY food-related query, ALWAYS call search_menu_catalog first — never ask
  clarifying questions without searching first. Show results, then offer to refine.
- Use conversation history to understand context. If the customer previously
  asked about a cuisine or preference, carry that forward even if the latest
  message is vague (e.g. "anything works" after asking about non-veg → search
  for non-veg dishes).
- Respond in a warm, helpful tone. Mention dietary tags proactively.
- Keep responses concise — this is a voice assistant.
"""

menu_llm = llm.bind_tools(menu_tools_list)

def menu_agent_node(state: StackState) -> dict:
    """Call the LLM. On first invocation build the initial prompt
    using persisted conversation history; on subsequent invocations
    (after tool results) continue from menu_messages."""

    existing = state.get("menu_messages") or []

    if not existing:
        # First call — conversation history is persisted via MemorySaver
        history = state.get("messages", [])[-6:]

        all_msgs = [
            SystemMessage(content=MENU_AGENT_PROMPT),
            *history,
            HumanMessage(content=state["user_query"]),
        ]
    else:
        # Subsequent call — tool results already in menu_messages
        all_msgs = existing

    response = menu_llm.invoke(all_msgs)
    all_msgs = [*all_msgs, response]

    has_tools = bool(getattr(response, "tool_calls", None))

    update: dict = {"menu_messages": all_msgs}

    if not has_tools:
        update["menu_response"] = response.content
        update["messages"] = [response]

    return update