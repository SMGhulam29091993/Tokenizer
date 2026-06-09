from google import genai
from google.genai import types
from dotenv import load_dotenv
import json

load_dotenv()
client = genai.Client()

model = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You should only answer coding, programming and software development related questions. "
    "If the question is not related to coding, programming or software development, "
    "you should respond with 'I am sorry, I can only answer coding, programming and software development related questions.' "
    "Your name is Jarvis."
)

response = client.models.generate_content(
    model=model,
    contents="Can you write a python code to find what is the capital of France?",
    config=types.GenerateContentConfig(
        system_instruction=(
            SYSTEM_INSTRUCTION
        )
    )
)

# print(json.dumps({"response": response.text}, indent=2))
print(response.text)
