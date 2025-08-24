# Make sure to install OpenAI SDK: `pip3 install openai`
from openai import OpenAI

# Remove angle brackets around the API key
client = OpenAI(
    api_key="sk-66ff70d8e15244dfb4bf8e91b63085b4",  # 🔑 Key WITHOUT < >
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)