import json
import os
import httpx

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal


class SupportRequest(BaseModel):
    intent: Literal[
        "order_status",
        "refund",
        "cancel_order",
        "technical_issue",
        "other",
    ]

    urgency: Literal[
        "low",
        "normal",
        "high",
    ]

    order_id: str | None = Field(
        default=None,
        description="The customer's order ID, if provided.",
    )


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(),
)


system_prompt = """
You classify customer support requests.

Return JSON only.

The JSON must have exactly these fields:

{
  "intent": "order_status | refund | cancel_order | technical_issue | other",
  "urgency": "low | normal | high",
  "order_id": "string or null"
}
"""


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "My order 12345 hasn't arrived and I'm really "
                "angry about it."
            ),
        },
    ],
    response_format={
        "type": "json_object",
    },
)

raw_content = response.choices[0].message.content

data = json.loads(raw_content)

support_request = SupportRequest.model_validate(data)

print(support_request)
print()
print("Intent:", support_request.intent)
print("Urgency:", support_request.urgency)
print("Order:", support_request.order_id)