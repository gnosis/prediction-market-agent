import asyncio

import streamlit as st
from autogen import Cache

from prediction_market_agent.agents.learnable_agent.agents import (
    get_code_writer_agent,
    get_code_executor_agent,
)
from prediction_market_agent.agents.learnable_agent.functions import (
    register_all_functions,
)

st.write("""# AutoGen Chat Agents""")


with st.container():
    #    st.markdown(message)
    st.markdown(
        """Use the web3.py Python library and interact with the Conditional Tokens contract on the Gnosis Chain (contract address 0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce) in order to read the balance of wallet address 0x2FC96c4e7818dBdc3D72A463F47f0E1CeEa0A2D0 with position id 38804060408381130621475891941405037249059836800475827360004002125093421139610.
                    Return the balance fetched using the latest block.
                    Consider using the function execute_read_function to execute a read function on the smart contract.
                    Whenever passing an address as parameter, calculate the checksum address of the address.
                    Let's think step-by-step.
                    """
    )
    user_input = st.chat_input(
        placeholder="Copy prompt from above",
    )
    if user_input:
        code_executor_agent = get_code_executor_agent(for_streamlit=True)
        code_writer_agent = get_code_writer_agent(for_streamlit=True)

        register_all_functions(
            caller_agent=code_writer_agent, executor_agent=code_executor_agent
        )

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Define an asynchronous function
        async def initiate_chat():
            with Cache.disk(cache_path_root="/tmp/autogen_cache_app2") as cache:
                await code_executor_agent.a_initiate_chat(
                    code_writer_agent,
                    message=user_input,
                    cache=cache,
                )
                # The correct result should be 289421015806737773 (as of block 37141392)
                # print(chat_result)

                # await code_executor_agent.a_initiate_chat(
                #     assistant,
                #     message=user_input,
                # )

        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
