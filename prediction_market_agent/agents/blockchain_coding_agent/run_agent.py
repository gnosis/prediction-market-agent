from prediction_market_agent.agents.blockchain_coding_agent.agents import (
    get_code_writer_agent,
    get_code_rag_agent,
)
from prediction_market_agent.agents.blockchain_coding_agent.functions import (
    register_all_functions,
)

if __name__ == "__main__":
    # code_executor_agent = get_code_executor_agent(for_streamlit=False)
    code_writer_agent = get_code_writer_agent(for_streamlit=False)
    rag_agent = get_code_rag_agent(for_streamlit=False)

    # register_all_functions(
    #     caller_agent=code_writer_agent, executor_agent=code_executor_agent
    # )
    register_all_functions(caller_agent=code_writer_agent, executor_agent=rag_agent)

    message_read = "Use the web3.py Python library and interact with the Conditional Tokens contract on the Gnosis Chain (contract address 0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce) in order to read the balance of wallet address 0x2FC96c4e7818dBdc3D72A463F47f0E1CeEa0A2D0 with position id 38804060408381130621475891941405037249059836800475827360004002125093421139610. Return the balance fetched using the latest block. Consider using the function execute_read_function to execute a read function on the smart contract. Whenever passing an address as parameter, calculate the checksum address of the address. Let's think step-by-step."

    message_write = "Use the web3.py Python library and the registered functions and interact with the USDC token contract on the Gnosis Chain (contract address 0xddafbb505ad214d7b80b1f830fccc89b60fb7a83). Approve Johnny (wallet address 0x70997970C51812dc3A010C7d01b50e0d17dc79C8) as spender of your USDC, with allowance 100 USDC. Let's think step-by-step and make use of web3py functions."

    # groupchat = GroupChat(
    #     agents=[code_writer_agent, rag_agent, code_executor_agent],
    #     messages=[],
    #     max_round=12,
    #     speaker_selection_method="round_robin",
    #     allow_repeat_speaker=False,
    # )

    # config_list = [
    #     {"model": "gpt-4o-mini", "api_key": APIKeys().openai_api_key.get_secret_value()}
    # ]
    # llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0}
    # manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # with Cache.disk(cache_path_root="/tmp/autogen_cache_run_agent") as cache:
    chat_result = rag_agent.initiate_chat(
        code_writer_agent, message=rag_agent.message_generator, problem=message_write
    )
    # code_executor_agent.initiate_chat(
    #    code_writer_agent, message=message_write, cache=cache, max_turns=30
    # )
    print("finished")
