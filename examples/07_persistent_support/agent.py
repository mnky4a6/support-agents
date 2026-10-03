import os
from pathlib import Path

import httpx
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
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


builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([get_order]))
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

Path("data").mkdir(exist_ok=True)

with SqliteSaver.from_conn_string(
    "data/support-agent.db"
) as checkpointer:

    graph = builder.compile(
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread_id": "customer-123"
        }
    }

    '''
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My order number is 12345.",
                }
            ]
        },
        config,
    )
    '''
    
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Where is it?",
                }
            ]
        },
        config,
    )

    print(result["messages"][-1].content)