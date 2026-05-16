from langgraph.prebuilt import ToolNode

from tools.order_tools import order_tools_list

tool_node = ToolNode(order_tools_list,messages_key="order_messages")

