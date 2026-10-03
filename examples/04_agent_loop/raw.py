import json
import os

import httpx
from openai import OpenAI
from pydantic import BaseModel, ValidationError


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


class GetOrderArguments(BaseModel):
    order_id: str


def get_order(order_id: str) -> dict:
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


TOOL_MAP = {
    "get_order": get_order,
}


TOOLS = [
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


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(),
)


def run_agent(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer support agent. "
                "Use tools when you need order information. "
                "Never invent order information."
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    while True:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=TOOLS,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )

        message = response.choices[0].message

        # Always preserve the assistant message in the conversation.
        messages.append(message)

        # No tool call means the agent is finished.
        if not message.tool_calls:
            return message.content or ""

        # Execute every tool requested by the model.
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            if tool_name not in TOOL_MAP:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            try:
                raw_arguments = json.loads(
                    tool_call.function.arguments
                )
                arguments = GetOrderArguments.model_validate(
                    raw_arguments
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid arguments for {tool_name}"
                ) from exc

            tool_function = TOOL_MAP[tool_name]

            result = tool_function(
                arguments.order_id
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    answer = run_agent(
        "Where is my order 12345?"
    )

    print(answer)