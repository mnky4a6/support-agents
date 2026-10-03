import os

import httpx
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek


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


model_with_tools = model.bind_tools(
    [get_order]
)


response = model_with_tools.invoke(
    "Where is my order 12345?"
)


print(response)
print()
print("Tool calls:")
print(response.tool_calls)