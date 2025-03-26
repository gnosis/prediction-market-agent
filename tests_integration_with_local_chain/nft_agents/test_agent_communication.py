import zlib

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentCommunicationContract,
    AgentRegisterContract,
)


def test_count_unseen_messages(local_web3: Web3, test_keys: APIKeys) -> None:
    # Use our address as recipient for simplicity.
    recipient = test_keys.bet_from_address
    comm_contract = AgentCommunicationContract()
    reg_contract = AgentRegisterContract()

    # It might be the case that initial_messages > 0 (due to ape's tests not being isolated).
    initial_messages = comm_contract.count_unseen_messages(
        agent_address=recipient, web3=local_web3
    )

    # register ourselves, otherwise we can't send the message to our address.
    reg_contract.register_as_agent(api_keys=test_keys, web3=local_web3)

    # add new message
    message = zlib.compress(b"Hello there!")
    comm_contract.send_message(
        api_keys=test_keys,
        agent_address=recipient,
        message=HexBytes(message),
        amount_wei=xDai(0.1).as_xdai_wei,
        web3=local_web3,
    )
    assert (
        comm_contract.count_unseen_messages(agent_address=recipient, web3=local_web3)
        == initial_messages + 1
    )


def test_pop_message(local_web3: Web3, test_keys: APIKeys) -> None:
    # Recipient same as caller since popMessage() will be called afterwards, and it requires
    # the sender to be the message popper.
    recipient = test_keys.bet_from_address
    print(
        f"balance {get_balances(address=test_keys.bet_from_address, web3=local_web3)}"
    )
    comm_contract = AgentCommunicationContract()
    reg_contract = AgentRegisterContract()

    # It might be the case that initial_messages > 0 (due to ape's tests not being isolated).
    initial_messages = comm_contract.count_unseen_messages(
        agent_address=recipient, web3=local_web3
    )

    # register ourselves, otherwise we can't send the message to our address.
    reg_contract.register_as_agent(api_keys=test_keys, web3=local_web3)

    # add new message
    message = zlib.compress(b"Hello there!")

    comm_contract.send_message(
        api_keys=test_keys,
        agent_address=recipient,
        message=HexBytes(message),
        amount_wei=xDai(0.1).as_xdai_wei,
        web3=local_web3,
    )
    assert (
        comm_contract.count_unseen_messages(agent_address=recipient, web3=local_web3)
        == initial_messages + 1
    )

    # get at index
    stored_message = comm_contract.get_at_index(
        agent_address=recipient, idx=0, web3=local_web3
    )
    print(f"stored message {stored_message}")
    # assert message match
    assert stored_message.recipient == recipient
    assert stored_message.message == HexBytes(message)

    # fetch latest message
    stored_message = comm_contract.pop_message(test_keys, web3=local_web3)
    # assert message match
    assert stored_message.recipient == recipient
    assert stored_message.message == message
