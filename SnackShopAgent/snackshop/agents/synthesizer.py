from langchain_core.messages import SystemMessage, HumanMessage

from config import llm
from state import StackState

SYNTHESIZER_PROMPT = """\
You are the response synthesizer for SnackShop, a voice-enabled food delivery assistant.
Combine the responses from specialist agents into a single, coherent, friendly reply.
Keep it concise and conversational — this will be spoken aloud by TTS, so avoid
markdown formatting like **, bullet points, or numbered lists. Use natural speech
phrasing instead.
If only one agent responded, just clean up and present that response.
"""

def synthesizer_node(state: StackState) -> dict:
    """Combine agent outputs into a final, friendly answer."""

    menu_resp = state.get("menu_response", "")
    order_resp = state.get("order_response", "")

    combined = ""
    if menu_resp:
        combined += f"[Menu Agent]\n{menu_resp}\n\n"
    if order_resp:
        combined += f"[Order Agent]\n{order_resp}\n"

    if not combined.strip():
        combined = "I could not find relevant information. Please try rephrasing your query."

    final = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(
                content=f"Customer query: {state['user_query']}\n\nAgent outputs:\n{combined}"
            ),
        ]
    )

    return {
        "final_answer": final.content,
        "messages": [final],
    }