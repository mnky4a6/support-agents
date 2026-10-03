import json
import os

import httpx
from openai import OpenAI


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


def get_order(order_id: str) -> dict:
    """Look up an order by ID."""

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


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(),
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID.",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]


messages = [
    {
        "role": "system",
        "content": (
            "You are a customer support assistant. "
            "Use get_order when the user asks about an order."
        ),
    },
    {
        "role": "user",
        "content": "Where is my order 12345?",
    },
]


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools,
)

message = response.choices[0].message

print("FIRST MODEL RESPONSE")
print(message)

messages.append(message)

for tool_call in message.tool_calls:
    if tool_call.function.name != "get_order":
        raise ValueError(
            f"Unknown tool: {tool_call.function.name}"
        )

    arguments = json.loads(tool_call.function.arguments)

    order_id = arguments["order_id"]

    result = get_order(order_id)

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }
    )

    response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools,
)

print(response.choices[0].message.content)