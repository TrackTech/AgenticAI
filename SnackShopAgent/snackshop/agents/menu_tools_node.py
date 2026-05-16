from langgraph.prebuilt import ToolNode

from tools.menu_tools import menu_tools_list

# 1. Define your tool list

"""
    Original implementation 
    def menu_tools_node(state: StackState) -> dict:

    all_msgs = list(state["menu_messages"])
    last_msg = all_msgs[-1]
    tool_names = [tc["name"] for tc in last_msg.tool_calls]
    logger.info("Executing tools: %s", tool_names)

    for tc in last_msg.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        all_msgs.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"menu_messages": all_msgs}

    ToolNode replaces this but it looks for a state["message"] as key, instead of menu_messages. 
    """


tool_node = ToolNode(menu_tools_list,messages_key="menu_messages")