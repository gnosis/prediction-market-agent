# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
import tempfile
from pathlib import Path
from typing import Annotated

from autogen import ConversableAgent, Cache, register_function
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor
from tavily import TavilyClient

from prediction_market_agent.utils import APIKeys
from scripts.web3_scan_utils import (
    fetch_read_methods_from_blockscout,
    get_rpc_endpoint,
    execute_read_function,
    checksum_address,
)

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
# executor = LocalCommandLineCodeExecutor(
#     timeout=10,  # Timeout for each code execution in seconds.
#     work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
# )


server = LocalJupyterServer()


# executor = DockerCommandLineCodeExecutor(
#     image="python:3.12-slim",  # Execute code using the given docker image name.
#     timeout=10,  # Timeout for each code execution in seconds.
#     work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
# )

output_dir = Path("coding")
output_dir.mkdir(exist_ok=True)

tavily = TavilyClient(api_key=APIKeys().tavily_api_key.get_secret_value())


def search_tool(
    query: Annotated[str, "The search query"]
) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")


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

# Register the search tool.
register_function(
    search_tool,
    caller=code_writer_agent,
    executor=code_executor_agent,
    name="search_tool",
    description="Search the web for the given query",
)

# Register the calculator function to the two agents.
register_function(
    fetch_read_methods_from_blockscout,
    caller=code_writer_agent,  # The assistant agent can suggest calls to the calculator.
    executor=code_executor_agent,  # The user proxy agent can execute the calculator calls.
    name="fetch_read_methods_from_blockscout",  # By default, the function name is used as the tool name.
    description="Function for fetching the ABI from a verified smart contract on Gnosis Chain",  # A description of the tool.
)

register_function(
    get_rpc_endpoint,
    caller=code_writer_agent,  # The assistant agent can suggest calls to the calculator.
    executor=code_executor_agent,  # The user proxy agent can execute the calculator calls.
    name="get_rpc_endpoint",  # By default, the function name is used as the tool name.
    description="Returns the RPC endpoint to be used for interacting with Gnosis Chain when instantiating a provider",  # A description of the tool.
)

register_function(
    execute_read_function,
    caller=code_writer_agent,
    executor=code_executor_agent,
    name="execute_read_function",
    description="Executes a function on a smart contract deployed on the Gnosis Chain and returns the result of the function execution.",
)

register_function(
    checksum_address,
    caller=code_writer_agent,
    executor=code_executor_agent,
    name="checksum_address",
    description="Extracts the checksummed address of an address.",
)


with Cache.disk(cache_path_root="/tmp/autogen_cache2") as cache:
    chat_result = code_executor_agent.initiate_chat(
        code_writer_agent,
        message="""Use the web3.py Python library and interact with the Conditional Tokens contract on the Gnosis Chain (contract address 0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce) in order to read the balance of wallet address 0x2FC96c4e7818dBdc3D72A463F47f0E1CeEa0A2D0 with position id 38804060408381130621475891941405037249059836800475827360004002125093421139610. 
        Return the balance fetched using the latest block.
        Consider using the function execute_read_function to execute a read function on the smart contract.
        Whenever passing an address as parameter, calculate the checksum address of the address.
        Let's think step-by-step.
        """,
        cache=cache,
    )
    # The correct result should be 289421015806737773 (as of block 37141392)
    print(chat_result)
    # We
