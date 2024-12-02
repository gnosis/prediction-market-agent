# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
import tempfile
from pathlib import Path

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from prediction_market_agent.utils import APIKeys

code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""

keys = APIKeys()
temp_dir = tempfile.TemporaryDirectory()


# Create a local command line code executor.
# ToDo - venv
executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
)


server = LocalJupyterServer()

output_dir = Path("coding")
output_dir.mkdir(exist_ok=True)


class CodeOutput(BaseModel):
    code: str
    # dependencies: list[str]


parser = PydanticOutputParser(pydantic_object=CodeOutput)

code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={
        "executor": JupyterCodeExecutor(server, output_dir=output_dir)
    },  # Use the local command line code executor.
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message=code_writer_system_message,
    llm_config={
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": keys.openai_api_key.get_secret_value()}
        ]
    },
    code_execution_config=False,  # Turn off code execution for this agent.
)


"""
# ToDO
#  CodeWriter - writes step-by-step functions and nothing else (outputs BabyAGI Function BaseModel)
# Registers new functions
# Create new prompt (with all functions), asks it to execute code and returns output.
"""


for i in range(10):
    reply = code_writer_agent.generate_reply(
        messages=[
            {
                "role": "user",
                "content": """
        Create a fibonacci function. Output a function structure like below.
        
        [output instructions]
        {format_instructions}
        """.format(
                    format_instructions=parser.get_format_instructions()
                ),
            }
        ]
    )
    print("reply: ", reply)
