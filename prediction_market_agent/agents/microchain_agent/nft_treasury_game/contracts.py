import os
import typing as t
from contextlib import contextmanager

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import (
    ABI,
    ChecksumAddress,
    TxReceipt,
    Wei,
    xDai,
    xDaiWei,
)
from prediction_market_agent_tooling.tools.balances import Balances, get_balances
from prediction_market_agent_tooling.tools.contract import (
    ContractERC721BaseClass,
    ContractOnGnosisChain,
    ContractOwnableERC721OnGnosisChain,
    OwnableContract,
    abi_field_validator,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import BPS_CONSTANT
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.data_models import (
    MessageContainer,
)


class NFTKeysContract(ContractOwnableERC721OnGnosisChain):
    address: ChecksumAddress = Web3.to_checksum_address(
        "0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"
    )

    @staticmethod
    def retrieve_total_number_of_keys() -> int:
        # We could iteratively call `owner_of` for a range of token_ids, thus finding out the max supply. However,
        # in the current implementation, no new tokens can be created and max_supply = 5, hence hardcoding it here.
        return 5


class AgentRegisterContract(ContractOnGnosisChain, OwnableContract):
    # Contract ABI taken from built https://github.com/gnosis/labs-contracts.
    abi: ABI = abi_field_validator(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../abis/agentregistry.abi.json",
        )
    )

    address: ChecksumAddress = Web3.to_checksum_address(
        "0xe8ae78b19c997b6da8189b1a644d4076f8bc880e"
    )

    def register_as_agent(
        self,
        api_keys: APIKeys,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="registerAsAgent",
            web3=web3,
        )

    def deregister_as_agent(
        self,
        api_keys: APIKeys,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send(
            api_keys=api_keys,
            function_name="deregisterAsAgent",
            web3=web3,
        )

    def is_registered_agent(
        self,
        agent_address: ChecksumAddress,
        web3: Web3 | None = None,
    ) -> bool:
        is_registered_agent: bool = self.call(
            "isRegisteredAgent",
            function_params=[agent_address],
            web3=web3,
        )
        return is_registered_agent

    def get_all_registered_agents(
        self,
        web3: Web3 | None = None,
    ) -> t.List[ChecksumAddress]:
        return [
            Web3.to_checksum_address(addr)
            for addr in self.call("getAllRegisteredAgents", web3=web3)
        ]

    @contextmanager
    def with_unregistered_agent(
        self, api_keys: APIKeys, web3: Web3 | None = None
    ) -> t.Generator[None, None, None]:
        """
        Use this context manager to temporarily deregister the agent, and then re-register it after the block, if it was registered before.
        """
        was_registered = self.is_registered_agent(
            agent_address=api_keys.bet_from_address, web3=web3
        )
        if was_registered:
            self.deregister_as_agent(api_keys, web3=web3)
        yield
        if was_registered:
            self.register_as_agent(api_keys, web3=web3)

    @contextmanager
    def with_registered_agent(
        self, api_keys: APIKeys, web3: Web3 | None = None
    ) -> t.Generator[None, None, None]:
        """
        Use this context manager to temporarily register the agent, and then deregister it after the block, if it was not registered before.
        """
        was_registered = self.is_registered_agent(
            agent_address=api_keys.bet_from_address, web3=web3
        )
        if not was_registered:
            self.register_as_agent(api_keys, web3=web3)
        yield
        if not was_registered:
            self.deregister_as_agent(api_keys, web3=web3)


class AgentCommunicationContract(ContractOnGnosisChain, OwnableContract):
    # Contract ABI taken from built https://github.com/gnosis/labs-contracts.
    abi: ABI = abi_field_validator(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../abis/agentcommunication.abi.json",
        )
    )

    address: ChecksumAddress = Web3.to_checksum_address(
        "0xca6c43b46febb0505d13a7704084912883eecf32"
    )

    def minimum_message_value(self, web3: Web3 | None = None) -> xDai:
        value = xDaiWei(self.call("minimumValueForSendingMessageInWei", web3=web3))
        return value.as_xdai

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
            agent_address=api_keys.bet_from_address, idx=index, web3=web3
        )

        # Next, pop that element and discard the transaction receipt.
        self.send(
            api_keys=api_keys,
            function_name="popMessageAtIndex",
            function_params=[index],
            web3=web3,
        )

        return message_container

    def send_message(
        self,
        api_keys: APIKeys,
        agent_address: ChecksumAddress,
        message: HexBytes,
        amount_wei: xDaiWei,
        web3: Web3 | None = None,
    ) -> TxReceipt:
        return self.send_with_value(
            api_keys=api_keys,
            function_name="sendMessage",
            amount_wei=amount_wei.as_wei,
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

    def balances(self, web3: Web3 | None = None) -> Balances:
        return get_balances(self.address, web3=web3)

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
