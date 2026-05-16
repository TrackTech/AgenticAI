from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import StackState

from agents.orchestrator import orchestrator_node
from agents.menu_agent import menu_agent_node
from agents.menu_tools_node import tool_node as menu_tools_node
from agents.order_agent import order_agent_node
from agents.order_tools_node import tool_node as order_tools_node
from agents.synthesizer import synthesizer_node


def should_continue_menu(state: StackState) -> str:
    """Conditional edge: route to tools or synthesizer."""
    menu_msgs = state.get("menu_messages") or []
    if menu_msgs:
        last = menu_msgs[-1]
        if getattr(last, "tool_calls", None):
            return "menu_tools_node"
    return "synthesizer_node"

def should_continue_order(state: StackState) -> str:
    """Conditional edge: route to tools or synthesizer."""
    order_msgs = state.get("order_messages") or []
    if order_msgs:
        last = order_msgs[-1]
        if getattr(last, "tool_calls", None):
            return "order_tools_node"
    return "synthesizer_node"

def build_graph():
    """Construct, compile, and return the SnackStack graph."""

    builder = StateGraph(StackState)

    # ── Nodes ───────────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("menu_agent_node", menu_agent_node)
    builder.add_node("menu_tools_node", menu_tools_node)
    builder.add_node("order_agent_node", order_agent_node)
    builder.add_node("order_tools_node", order_tools_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    # ── Edges ───────────────────────────────────────────────

    # START → orchestrator (fixed)
    builder.add_edge(START, "orchestrator")
    # orchestrator → agent(s) routing handled via Command + Send()

    # Menu agent: conditional → tools (loop) or synthesizer
    builder.add_conditional_edges(
        "menu_agent_node",
        should_continue_menu,
        {"menu_tools_node": "menu_tools_node", "synthesizer_node": "synthesizer_node"},
    )
    builder.add_edge("menu_tools_node", "menu_agent_node")

    # Order agent: conditional → tools (loop) or synthesizer
    builder.add_conditional_edges(
        "order_agent_node",
        should_continue_order,
        {"order_tools_node": "order_tools_node", "synthesizer_node": "synthesizer_node"},
    )
    builder.add_edge("order_tools_node", "order_agent_node")

    # Synthesizer → END (fixed)
    builder.add_edge("synthesizer_node", END)

    # ── Compile with memory ─────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # logger.info("SnackStack graph compiled")
    return graph