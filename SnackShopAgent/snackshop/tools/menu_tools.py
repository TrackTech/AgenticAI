"""
tools/menu_tools.py — Tools available to the Menu Agent.
"""

from langchain_core.tools import tool
from tools.rag import menu_retriever
from logger import setup_logger
logger = setup_logger("snackshop.menu_tools")


@tool
def search_menu_catalog(query: str) -> str:
    """Search the SnackStack menu catalog using semantic similarity.

    Use this to find dishes by name, cuisine, dietary preference, or description.

    Args:
        query: Natural language search query, e.g. 'vegan pasta' or 'spicy Indian starter'

    Returns:
        Formatted list of matching menu items with details.
    """

    logger.info(f"Menu Retriver (RAG) query: {query}")
    results = menu_retriever.invoke(query)

    if not results:
        return "No matching dishes found for your query."

    output = f'Top {len(results)} matches for "{query}":\n\n'
    for doc in results:
        output += doc.page_content + "\n---\n"
    logger.info(f"Menu Retriever results:\n{output}")
    return output


# Convenient list for binding to the LLM / building ToolNodes
menu_tools_list = [search_menu_catalog]