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

from aea.protocols.base import Message
from aea.skills.base import Handler
from web3 import Web3
from web3.types import TxReceipt
from typing import Callable, cast
from packages.valory.protocols.websocket_client.message import WebsocketClientMessage
from enum import Enum

JOB_QUEUE = "pending_tasks"
SUBSCRIPTION_ID = "subscription_id"
WEBSOCKET_SUBSCRIPTION_STATUS = "websocket_subscription_status"
WEBSOCKET_SUBSCRIPTIONS = "websocket_subscriptions"


class SubscriptionStatus(Enum):
    """Subscription status."""

    UNSUBSCRIBED = "unsubscribed"
    SUBSCRIBING = "subscribing"
    CHECKING_SUBSCRIPTION = "checking_subscription"
    SUBSCRIBED = "subscribed"


class WebSocketHandler(Handler):
    """This class scaffolds a handler."""

    SUPPORTED_PROTOCOL = WebsocketClientMessage.protocol_id

    def setup(self) -> None:
        """Implement the setup."""
        if WEBSOCKET_SUBSCRIPTION_STATUS not in self.context.shared_state:
            self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS] = {}

        if WEBSOCKET_SUBSCRIPTIONS not in self.context.shared_state:
            self.context.shared_state[WEBSOCKET_SUBSCRIPTIONS] = {}

        self._count = 0

    def handle(self, message: WebsocketClientMessage) -> None:
        """
        Implement the reaction to an envelope.

        :param message: the message
        """
        self.context.logger.info(f"Received message: {message}")
        handler = cast(
            Callable[[WebsocketClientMessage], None],
            getattr(self, f"handle_{message.performative.value}"),
        )
        handler(message)

    def handle_subscription(self, message: WebsocketClientMessage) -> None:
        """Handler `WebsocketClientMessage.Performative.SUBSCRIPTION` response"""
        self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
            message.subscription_id
        ] = (
            SubscriptionStatus.SUBSCRIBED
            if message.alive
            else SubscriptionStatus.UNSUBSCRIBED
        )

    def handle_send_success(self, message: WebsocketClientMessage) -> None:
        """Handler `WebsocketClientMessage.Performative.SEND_SUCCESS` response"""
        self.context.logger.info(
            f"Sent data to the websocket with id {message.subscription_id}; send_length: {message.send_length}"
        )

    def handle_recv(self, message: WebsocketClientMessage) -> None:
        """Handler `WebsocketClientMessage.Performative.RECV` response"""
        self.context.logger.info(
            f"Received {message.data} from subscription {message.subscription_id}"
        )
        subscription_id = message.subscription_id
        if subscription_id not in self.context.shared_state[WEBSOCKET_SUBSCRIPTIONS]:
            self.context.shared_state[WEBSOCKET_SUBSCRIPTIONS][subscription_id] = []

        self.context.shared_state[WEBSOCKET_SUBSCRIPTIONS][subscription_id].append(
            message.data
        )

    def handle_error(self, message: WebsocketClientMessage) -> None:
        """Handler `WebsocketClientMessage.Performative.ERROR` response"""
        self.context.logger.info(
            f"Error occured on the websocket with id {message.subscription_id}; Error: {message.message}"
        )
        self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
            message.subscription_id
        ] = (
            SubscriptionStatus.SUBSCRIBED
            if message.alive
            else SubscriptionStatus.UNSUBSCRIBED
        )

    def teardown(self) -> None:
        """Implement the handler teardown."""
