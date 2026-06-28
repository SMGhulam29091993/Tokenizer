from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import json
import subprocess
import time

load_dotenv()

client = genai.Client()

model = "gemini-2.5-flash"

def run_command(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

available_tools = {
    "run_command" : run_command
}

SYSTEM_PROMPT="""
 You're an expert AI Assisstant in resolving user queries using chain of thought.
 You work on START, PLAN, and OUTPUT steps.
 You need to first plan what needs to be done. The plan can be of multiple steps.
 Once you think enough plan has been done, finally you can give an OUTPUT.
 You can also call a tool if required from the list of available tools.
 For every tool call wait for the observe step which is the ouptut from the called tool.

 Rules:
 - Strictly Follow the given JSON output format
 - Only run one step at a time.
 - The sequence of steps is START (where user gives an input), PLAN (can be multiple times), and finally OUTPUT.
 - NEVER ask the user clarifying questions in PLAN steps. Make reasonable assumptions and proceed.
 - If tech stack is not specified, default to HTML/CSS/JavaScript (vanilla).
 - Always create a folder in the current working directory before writing any files.

 Output JSON Format:
 {"step" : "START", | "PLAN" | "OUTPUT" | "TOOL", "content" : "string", "tool" : "string", "input" : "string"}

 Available Tools: 
 - run_command(cmd : str): Takes a system linux command as a string and executes the command on the users system and returns the output from that command

 Example 1:
 START : Hey, Can you solve 2 + 3 * 5 / 10
 PLAN : {"step" : "PLAN" : "content" : "Seems like user is interested in math problem"}
 PLAN : {"step" : "PLAN" : "content" : "Looking at the problem, we should solve this using BODMAS method"}
 PLAN : {"step" : "PLAN" : "content" : "Yes, the BODMAS is the correct thing to be done here"}
 PLAN : {"step" : "PLAN" : "content" : "First we must multiply 3 * 5 whis is 15"}
 PLAN : {"step" : "PLAN" : "content" : "Now the new equation is 2 + 15 / 10 "}
 PLAN : {"step" : "PLAN" : "content" : "We must first perform division that is 15 / 10 = 1.5"}
 PLAN : {"step" : "PLAN" : "content" : "Now the new equation is 2 + 1.5"}
 PLAN : {"step" : "PLAN" : "content" : "Now finally let's perform the add 3.5"}
 PLAN : {"step" : "PLAN" : "content" : "Great, we have finally solved and letf with 3.5 as an answer"}
 OUTPUT : {"step" : "OUTPUT" : "content" : "3.5"}

 Example 2 (tool use):
 START : List all Python files in the current directory
 PLAN : {"step" : "PLAN", "content" : "I need to list .py files. I will use the run_command tool with 'find . -name *.py'"}
 TOOL : {"step" : "TOOL", "tool" : "run_command", "input" : "find . -name '*.py'"}
 OBSERVE : {"step" : "OBSERVE" : "tool" : "run_command", "output": "<p>code</p>"}
 PLAN : {"step" : "PLAN", "content" : "I now have the list of Python files from the observation."}
 OUTPUT : {"step" : "OUTPUT", "content" : "The Python files are: main.py, utils.py"}
"""

class MyOutputFormat(BaseModel):
    step : str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content : Optional[str] = Field(None, description="The optional string content for the step.")
    tool : Optional[str] = Field(None, description="The ID of the tool to call")
    input : Optional[str] = Field(None, description="Input params for the tool")

print("\n\n\n")

message_history = [
    {"role" : "system", "content" : SYSTEM_PROMPT},
]

user_query = input("👉🏻 ")
message_history.append({"role" :"user", "content" : user_query})

def build_contents(history: list[dict]) -> list[types.Content]:
    contents = []
    for msg in history:
        if msg["role"] == "system":
            continue
        role = "model" if msg["role"] in ("assistant", "developer") else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    return contents

MAX_ITERATIONS = 30
iteration = 0

while iteration < MAX_ITERATIONS:
    iteration += 1

    try:
        response = client.models.generate_content(
            model=model,
            contents=build_contents(message_history),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=MyOutputFormat,
            ),
        )
    except ClientError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            wait = 60
            print(f"⏳ Rate limit hit, retrying in {wait}s...")
            time.sleep(wait)
            iteration -= 1
            continue
        raise

    time.sleep(13)  # free tier: 5 RPM → 12s minimum gap between calls

    raw_result = response.text or ""
    message_history.append({"role": "assistant", "content": raw_result})

    parsed_result = MyOutputFormat.model_validate_json(raw_result)

    if parsed_result.step == "START":
        print("🔥", parsed_result.content)
        continue

    if parsed_result.step == "TOOL":
        tool_to_call = parsed_result.tool or ""
        tool_input = parsed_result.input or ""
        tool_response = available_tools[tool_to_call](tool_input)
        print(f"🛠️ : {tool_to_call} ({tool_input}) = {tool_response}")
        message_history.append({"role": "developer", "content": json.dumps(
            {"step": "OBSERVE", "tool": tool_to_call, "input": tool_input, "output": tool_response}
        )})
        continue

    if parsed_result.step == "PLAN":
        print("🧠", parsed_result.content)
        continue

    if parsed_result.step == "OUTPUT":
        print("🥳", parsed_result.content)
        break
else:
    print("⚠️ Reached max iterations without a final OUTPUT.")

print("\n\n\n")
