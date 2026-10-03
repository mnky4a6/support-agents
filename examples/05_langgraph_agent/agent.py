import os

import httpx
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


ORDERS = {
    "12345": {
        "status": "shipped",
        "carrier": "UPS",
        "tracking_number": "1Z999AA10123456784",
        "estimated_delivery": "2026-10-05",
    },
    "67890": {
        "status": "processing",
        "carrier": None,
        "tracking_number": None,
        "estimated_delivery": None,
    },
}


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by its order ID."""

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "found": False,
            "order_id": order_id,
        }

    return {
        "found": True,
        "order_id": order_id,
        **order,
    }


model = ChatDeepSeek(
    model="deepseek-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    http_client=httpx.Client(),
)

model_with_tools = model.bind_tools([get_order])


def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])

    return {
        "messages": [response],
    }


tool_node = ToolNode([get_order])


builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "agent")

graph = builder.compile()


result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Where is my order 12345?",
            }
        ]
    }
)


for message in result["messages"]:
    print(message)