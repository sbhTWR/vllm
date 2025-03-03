# SPDX-License-Identifier: Apache-2.0

import json
from time import sleep
from openai import OpenAI
import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
print(models)
model = models.data[0].id

# Request 1 
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }],
    model=model,
    user=json.dumps({"id": "test_id", "type": "append"})
)

print("1. ------ Chat completion results --------")
print(chat_completion)

sleep(10)

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "Who won the world series in 2021?"
    }, {
        "role": "assistant",
        "content": "The Atlanta Braves won the World Series in 2021."
    }, {
        "role": "user",
        "content": "Who won the world series in 2021?"
    }],
    model=model,
    user=json.dumps({"id": "test_id", "type": "append"})
)

print("2. ------ Chat completion results --------")
print(chat_completion)

sleep(10)


try:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": "placeholder"
        }],
        model=model,
        user=json.dumps({"id": "test_id", "type": "fin"}),
        timeout=1
    )
except openai.APITimeoutError as e:
    print("3. ------ END --------")


# # Request 2 
# chat_completion = client.chat.completions.create(
#     messages=[{
#         "role": "user",
#         "content": "Where was it played?"
#     }],
#     model=model,
#     user=json.dumps({"id": "test_id", "type": "append"})
# )

