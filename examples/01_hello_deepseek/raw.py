import os
import httpx
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(),
)


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": "You are a concise customer support assistant.",
        },
        {
            "role": "user",
            "content": "My order has not arrived yet. What should I do?",
        },
    ],
)


#print(response.choices[0].message.content)
#print(response)
message = response.choices[0].message

print("Response:")
print(message.content)

print("\nUsage:")
print(f"Input tokens:  {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens:   {response.usage.total_tokens}")
