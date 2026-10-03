import os

import httpx
from pydantic import BaseModel, Field
from typing import Literal

from langchain_deepseek import ChatDeepSeek


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


model = ChatDeepSeek(
    model="deepseek-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    http_client=httpx.Client(),
)


structured_model = model.with_structured_output(
    SupportRequest,
    method="json_mode",
)


result = structured_model.invoke(
    """
    Classify this customer support request.

    The response must be JSON.
    
    The JSON must have exactly these fields:

    {
    "intent": "order_status | refund | cancel_order | technical_issue | other",
    "urgency": "low | normal | high",
    "order_id": "string or null"
    }
    
    Customer:
    I cannot log into the mobile application.
    

    """
    #I want my money back for order ABC-999.
    #    My order 12345 hasn't arrived and I'm really angry about it.

)


print(result)
print()
print("Intent:", result.intent)
print("Urgency:", result.urgency)
print("Order:", result.order_id)