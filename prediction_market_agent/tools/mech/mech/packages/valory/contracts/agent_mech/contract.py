# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the dynamic_contribution contract definition."""
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi
from aea_ledger_ethereum import EthereumApi
from web3 import Web3
from web3.types import BlockIdentifier, TxReceipt


partial_abis = [
    [
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "requestId",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "bytes",
                    "name": "data",
                    "type": "bytes",
                },
            ],
            "name": "Deliver",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": True,
                    "internalType": "address",
                    "name": "sender",
                    "type": "address",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "requestId",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "bytes",
                    "name": "data",
                    "type": "bytes",
                },
            ],
            "name": "Request",
            "type": "event",
        },
    ],
    [
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": True,
                    "internalType": "address",
                    "name": "sender",
                    "type": "address",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "requestId",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "requestIdWithNonce",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "bytes",
                    "name": "data",
                    "type": "bytes",
                },
            ],
            "name": "Request",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": True,
                    "internalType": "address",
                    "name": "sender",
                    "type": "address",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "requestId",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "bytes",
                    "name": "data",
                    "type": "bytes",
                },
            ],
            "name": "Deliver",
            "type": "event",
        },
    ],
]


class MechOperation(Enum):
    """Operation types."""

    CALL = 0
    DELEGATE_CALL = 1


class AgentMechContract(Contract):
    """The scaffold contract class for a smart contract."""

    contract_id = PublicId.from_str("valory/agent_mech:0.1.0")

    @classmethod
    def get_raw_transaction(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> JSONLike:
        """
        Handler method for the 'GET_RAW_TRANSACTION' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_raw_message(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> bytes:
        """
        Handler method for the 'GET_RAW_MESSAGE' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_state(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> JSONLike:
        """
        Handler method for the 'GET_STATE' requests.

        Implement this method in the sub class if you want
        to handle the contract requests manually.

        :param ledger_api: the ledger apis.
        :param contract_address: the contract address.
        :param kwargs: the keyword arguments.
        :return: the tx  # noqa: DAR202
        """
        raise NotImplementedError

    @classmethod
    def get_deliver_data(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        request_id: int,
        data: str,
        request_id_nonce: Optional[int],
    ) -> JSONLike:
        """
        Deliver a response to a request.

        :param ledger_api: LedgerApi object
        :param contract_address: the address of the token to be used
        :param request_id: the id of the target request
        :param data: the response data
        :param request_id_nonce: request id with nonce, to ensure uniqueness on-chain.
        :return: the deliver data
        """
        ledger_api = cast(EthereumApi, ledger_api)

        if not isinstance(ledger_api, EthereumApi):
            raise ValueError(f"Only EthereumApi is supported, got {type(ledger_api)}")

        deliver_with_nonce = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "requestId", "type": "uint256"},
                    {
                        "internalType": "uint256",
                        "name": "requestIdWithNonce",
                        "type": "uint256",
                    },
                    {"internalType": "bytes", "name": "data", "type": "bytes"},
                ],
                "name": "deliver",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            }
        ]
        if request_id_nonce is not None:
            contract_instance = ledger_api.api.eth.contract(
                contract_address, abi=deliver_with_nonce
            )
            data = contract_instance.encodeABI(
                fn_name="deliver",
                args=[request_id, request_id_nonce, bytes.fromhex(data)],
            )
        else:
            contract_instance = cls.get_instance(ledger_api, contract_address)
            data = contract_instance.encodeABI(
                fn_name="deliver", args=[request_id, bytes.fromhex(data)]
            )
        return {"data": bytes.fromhex(data[2:])}  # type: ignore

    @classmethod
    def get_request_events(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get the Request events emitted by the contract."""
        ledger_api = cast(EthereumApi, ledger_api)
        all_entries = []
        for abi in partial_abis:
            contract_instance = ledger_api.api.eth.contract(contract_address, abi=abi)
            entries = contract_instance.events.Request.create_filter(
                fromBlock=from_block,
                toBlock=to_block,
            ).get_all_entries()
            all_entries.extend(entries)

        request_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
                "contract_address": contract_address,
            }
            for entry in all_entries
        )
        return {"data": request_events}

    @classmethod
    def get_deliver_events(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get the Deliver events emitted by the contract."""
        ledger_api = cast(EthereumApi, ledger_api)
        all_entries = []
        for abi in partial_abis:
            contract_instance = ledger_api.api.eth.contract(contract_address, abi=abi)
            entries = contract_instance.events.Deliver.create_filter(
                fromBlock=from_block,
                toBlock=to_block,
            ).get_all_entries()
            all_entries.extend(entries)

        deliver_events = list(
            {
                "tx_hash": entry.transactionHash.hex(),
                "block_number": entry.blockNumber,
                **entry["args"],
            }
            for entry in all_entries
        )
        return {"data": deliver_events}

    @classmethod
    def process_tx_receipt(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        tx_receipt: TxReceipt,
    ) -> JSONLike:
        """Process transaction receipt to filter contract events."""

        ledger_api = cast(EthereumApi, ledger_api)
        contract_instance = cls.get_instance(ledger_api, contract_address)
        event, *_ = contract_instance.events.Request().processReceipt(tx_receipt)
        return dict(event["args"])

    @classmethod
    def get_undelivered_reqs(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
        max_block_window: int = 1000,
        **kwargs: Any,
    ) -> JSONLike:
        """Get the requests that are not delivered."""
        if from_block == "earliest":
            from_block = 0

        current_block = ledger_api.api.eth.block_number
        requests, delivers = [], []
        for from_block_batch in range(int(from_block), current_block, max_block_window):
            to_block_batch = (from_block_batch + max_block_window) - 1
            if to_block_batch >= current_block:
                to_block_batch = "latest"
            requests_batch: List[Dict[str, Any]] = cls.get_request_events(
                ledger_api, contract_address, from_block_batch, to_block_batch
            )["data"]
            delivers_batch: List[Dict[str, Any]] = cls.get_deliver_events(
                ledger_api, contract_address, from_block_batch, to_block_batch
            )["data"]
            requests.extend(requests_batch)
            delivers.extend(delivers_batch)
        pending_tasks: List[Dict[str, Any]] = []
        for request in requests:
            if request["requestId"] not in [
                deliver["requestId"] for deliver in delivers
            ]:
                # store each requests in the pending_tasks list, make sure each req is stored once
                pending_tasks.append(request)
        return {"data": pending_tasks}

    @classmethod
    def get_multiple_undelivered_reqs(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        contract_addresses: List[str],
        from_block: BlockIdentifier = "earliest",
        max_block_window: int = 1000,
        **kwargs: Any,
    ) -> JSONLike:
        """Get the requests that are not delivered."""
        pending_tasks: List[Dict[str, Any]] = []
        for contract_address in contract_addresses:
            pending_tasks_batch = cls.get_undelivered_reqs(
                ledger_api, contract_address, from_block, max_block_window
            ).get("data")
            pending_tasks.extend(pending_tasks_batch)
        return {"data": pending_tasks}

    @classmethod
    def get_exec_tx_data(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        to: str,
        value: int,
        data: bytes,
        operation: int,
        tx_gas: int,
    ) -> JSONLike:
        """Get tx data"""
        ledger_api = cast(EthereumApi, ledger_api)

        if not isinstance(ledger_api, EthereumApi):
            raise ValueError(f"Only EthereumApi is supported, got {type(ledger_api)}")

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="exec", args=[to, value, data, operation, tx_gas]
        )
        return {"data": bytes.fromhex(data[2:])}  # type: ignore

    @classmethod
    def get_subscription(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
    ) -> JSONLike:
        """Get tx data"""
        contract_instance = cls.get_instance(ledger_api, contract_address)
        nft = contract_instance.functions.subscriptionNFT().call()
        token_id = contract_instance.functions.subscriptionTokenId().call()
        return {"nft": nft, "token_id": token_id}

    @classmethod
    def get_set_subscription_tx_data(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        subscription_address: str,
        token_id: int,
    ) -> JSONLike:
        """Get tx data"""
        ledger_api = cast(EthereumApi, ledger_api)

        if not isinstance(ledger_api, EthereumApi):
            raise ValueError(f"Only EthereumApi is supported, got {type(ledger_api)}")

        contract_instance = cls.get_instance(ledger_api, contract_address)
        data = contract_instance.encodeABI(
            fn_name="setSubscription",
            args=[Web3.to_checksum_address(subscription_address), token_id],
        )
        return {"data": bytes.fromhex(data[2:])}  # type: ignore
