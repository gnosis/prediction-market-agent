# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#   Copyright 2023 eightballer
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

"""This package contains a scaffold of a handler."""

import json
import time
from typing import Any

from web3 import Web3
from web3.types import TxReceipt

from packages.valory.protocols.websocket_client.message import WebsocketClientMessage
from packages.valory.skills.websocket_client.handlers import (
    SubscriptionStatus,
    WEBSOCKET_SUBSCRIPTION_STATUS,
)
from packages.valory.skills.websocket_client.handlers import (
    WebSocketHandler as BaseWebSocketHandler,
)


JOB_QUEUE = "pending_tasks"
DISCONNECTION_POINT = "disconnection_point"


class WebSocketHandler(BaseWebSocketHandler):
    """This class scaffolds a handler."""

    SUPPORTED_PROTOCOL = WebsocketClientMessage.protocol_id
    w3: Web3 = None
    contract = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the handler."""
        self.websocket_provider = kwargs.pop("websocket_provider")
        self.contract_to_monitor = kwargs.pop("contract_to_monitor")
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""
        super().setup()

        self.context.shared_state[JOB_QUEUE] = []
        self.context.shared_state[DISCONNECTION_POINT] = None
        self._last_processed_block = None

        # loads the contracts from the config file
        with open(
            "vendor/valory/contracts/agent_mech/build/AgentMech.json",
            "r",
            encoding="utf-8",
        ) as file:
            abi = json.load(file)["abi"]

        self.w3 = Web3(  # pylint: disable=C0103
            Web3.HTTPProvider(self.websocket_provider)
        )
        self.contract = self.w3.eth.contract(address=self.contract_to_monitor, abi=abi)

    def handle(self, message: WebsocketClientMessage) -> None:
        """Handle message."""
        super().handle(message)
        if self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
            message.subscription_id
        ] in (SubscriptionStatus.UNSUBSCRIBED, SubscriptionStatus.SUBSCRIBING):
            self.context.logger.info(
                f"Setting disconnection point to {self._last_processed_block}"
            )
            self.context.shared_state[DISCONNECTION_POINT] = self._last_processed_block

    def handle_recv(self, message: WebsocketClientMessage) -> None:
        """Handler `RECV` performative"""
        try:
            data = json.loads(message.data)
        except json.JSONDecodeError:
            self.context.logger.info(
                f"Error decoding data for websocket subscription {message.subscription_id}; data={message.data}"
            )
            self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
                message.subscription_id
            ] = SubscriptionStatus.UNSUBSCRIBED
            return

        self.context.logger.info(
            f"Received {data} from subscription {message.subscription_id}"
        )

        if set(data.keys()) == {"id", "result", "jsonrpc"}:
            self.context.logger.info(f"Received response: {data}")
            return

        self.context.logger.info("Extracting data")
        tx_hash = data["params"]["result"]["transactionHash"]
        no_args = True
        limit = 0
        while no_args and limit < 10:
            event_args, no_request = self._get_tx_args(tx_hash)
            if no_request:
                self.context.logger.info("Event not a Request.")
                break
            if len(event_args) == 0:
                self.context.logger.info(f"Could not get event args. tx_hash={tx_hash}")
                time.sleep(1)
                limit += 1
                return
            no_args = False

        if len(event_args) != 0:
            self.context.shared_state[JOB_QUEUE].append(event_args)
            self.context.logger.info(f"Added job to queue: {event_args}")

    def _get_tx_args(self, tx_hash: str) -> Any:
        """Get the transaction arguments."""
        try:
            tx_receipt: TxReceipt = self.w3.eth.get_transaction_receipt(tx_hash)
            self._last_processed_block = tx_receipt["blockNumber"]
            rich_logs = self.contract.events.Request().processReceipt(tx_receipt)  # type: ignore
            return dict(rich_logs[0]["args"]), False

        except Exception as exc:  # pylint: disable=W0718
            self.context.logger.error(
                f"An exception occurred while trying to get the transaction arguments for {tx_hash}: {exc}"
            )
            return {}, True
