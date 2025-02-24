import os
import typing as t

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import (
    ABI,
    ChecksumAddress,
    TxReceipt,
    Wei,
    xDai,
)
from prediction_market_agent_tooling.tools.contract import (
    ContractERC721BaseClass,
    ContractOnGnosisChain,
    OwnableContract,
    abi_field_validator,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import BPS_CONSTANT
from prediction_market_agent_tooling.tools.web3_utils import wei_to_xdai
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.data_models import (
    MessageContainer,
)


class AgentCommunicationContract(ContractOnGnosisChain, OwnableContract):
    # Contract ABI taken from built https://github.com/gnosis/labs-contracts.
    abi: ABI = abi_field_validator(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../abis/agentcommunication.abi.json",
        )
    )

    address: ChecksumAddress = Web3.to_checksum_address(
        "0xe9dd78FF297DbaAbe5D0E45aE554a4B561935DE9"
    )

    def minimum_message_value(self, web3: Web3 | None = None) -> xDai:
        value: Wei = self.call("minimumValueForSendingMessageInWei", web3=web3)
        return wei_to_xdai(value)

    def ratio_given_to_treasury(self, web3: Web3 | None = None) -> float:
        bps: int = self.call("pctToTreasuryInBasisPoints", web3=web3)
        return bps / BPS_CONSTANT

    def count_unseen_messages(
        self,
        agent_address: ChecksumAddress,
        web3: Web3 | None = None,
    ) -> int:
        unseen_message_count: int = self.call(
            "countMessages", function_params=[agent_address], web3=web3
        )
        return unseen_message_count

    def get_at_index(
        self,
        agent_address: ChecksumAddress,
        idx: int,
        web3: Web3 | None = None,
    ) -> MessageContainer:
        message_container_raw: t.Tuple[t.Any] = self.call(
            "getAtIndex", function_params=[agent_address, idx], web3=web3
        )
        return MessageContainer.from_tuple(message_container_raw)

    def set_treasury_address(
        self,
        api_keys: APIKeys,
        new_treasury_address: ChecksumAddress,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="setTreasuryAddress",
            function_params=[new_treasury_address],
            web3=web3,
        )

    def set_minimum_value_for_sending_message(
        self,
        api_keys: APIKeys,
        new_minimum_value: Wei,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="adjustMinimumValueForSendingMessage",
            function_params=[new_minimum_value],
            web3=web3,
        )

    def pop_message(
        self,
        api_keys: APIKeys,
        agent_address: ChecksumAddress,
        index: int = 0,
        web3: Web3 | None = None,
    ) -> MessageContainer:
        """
        Retrieves and removes message at specified index from the agent's message queue.

        This method first retrieves the message at the front of the queue without removing it,
        allowing us to return the message content directly. The actual removal of the message
        from the queue is performed by sending a transaction to the contract, which executes
        the `popMessageAtIndex` function. The transaction receipt is not used to obtain the message
        content, as it only contains event data, not the returned struct.
        """

        # Peek the element before popping.
        message_container = self.get_at_index(
            agent_address=agent_address, idx=index, web3=web3
        )

        # Next, pop that element and discard the transaction receipt.
        self.send(
            api_keys=api_keys,
            function_name="popMessageAtIndex",
            function_params=[agent_address, index],
            web3=web3,
        )

        return message_container

    def send_message(
        self,
        api_keys: APIKeys,
        agent_address: ChecksumAddress,
        message: HexBytes,
        amount_wei: Wei,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send_with_value(
            api_keys=api_keys,
            function_name="sendMessage",
            amount_wei=amount_wei,
            function_params=[agent_address, message],
            web3=web3,
        )


class SimpleTreasuryContract(ContractOnGnosisChain, OwnableContract):
    # Contract ABI taken from built https://github.com/gnosis/labs-contracts.
    abi: ABI = abi_field_validator(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../abis/simpletreasury.abi.json",
        )
    )

    address: ChecksumAddress = Web3.to_checksum_address(
        "0x624ad0db52e6b18afb4d36b8e79d0c2a74f3fc8a"
    )

    def required_nft_balance(self, web3: Web3 | None = None) -> int:
        min_num_of_nfts: int = self.call("requiredNFTBalance", web3=web3)
        return min_num_of_nfts

    def set_required_nft_balance(
        self,
        api_keys: APIKeys,
        new_required_balance: int,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="setRequiredNFTBalance",
            function_params=[new_required_balance],
            web3=web3,
        )

    def nft_contract(self, web3: Web3 | None = None) -> ContractERC721BaseClass:
        nft_contract_address: ChecksumAddress = self.call("nftContract", web3=web3)
        contract = ContractERC721BaseClass(address=nft_contract_address)
        return contract

    def withdraw(
        self,
        api_keys: APIKeys,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="withdraw",
            web3=web3,
        )
