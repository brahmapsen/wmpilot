from openai import OpenAI
from dotenv import load_dotenv
import os, time

load_dotenv()
aiml_api_key = os.getenv("AIML_API_KEY")
if not aiml_api_key:
    raise ValueError("AIML_API_KEY not found in .env file")

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key= aiml_api_key,
)

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}]
# )
# print(response.choices[0].message.content)

system_prompt = """
You are primary care physician who can diagonose patient symptoms.

The diagnosis
- regurgitate the request back so that user knows that you are replying to his query.
- suggests an answer
- Ask at least one and up to 3 follow up questions for missing adequate information situation.

Temper your suggestion with a disclaimer that you are an AI assistant NOT a real physician.
"""


def generate_response(system_prompt, user_prompt, model="openai/gpt-5-chat-latest"):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
                ]
    )
    print(f"{time.time() - start_time:.2f} seconds")
    return response.choices[0].message.content

user_prompt = "I feel tightness in my chest after jogging only 50 steps or so."
response = generate_response(system_prompt, user_prompt)

print(f"{response}")
