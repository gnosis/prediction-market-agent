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

"""This package contains a scaffold of a behaviour."""

import json
import re
from abc import ABC
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Set, Type, cast

from aea.mail.base import Envelope
from aea.skills.behaviours import SimpleBehaviour

from packages.valory.connections.websocket_client.connection import \
    PUBLIC_ID as WEBSOCKET_CLIENT_CONNECTION
from packages.valory.connections.websocket_client.connection import \
    WebSocketClient
from packages.valory.protocols.websocket_client.message import \
    WebsocketClientMessage
from packages.valory.skills.websocket_client.dialogues import (
    WebsocketClientDialogue, WebsocketClientDialogues)
from packages.valory.skills.websocket_client.handlers import (
    WEBSOCKET_SUBSCRIPTION_STATUS, WEBSOCKET_SUBSCRIPTIONS, SubscriptionStatus)
from packages.valory.skills.websocket_client.models import Params

DEFAULT_ENCODING = "utf-8"
WEBSOCKET_CLIENT_CONNECTION_NAME = "websocket_client"


class SubscriptionBehaviour(SimpleBehaviour):
    """This class scaffolds a behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the agent."""
        self._contracts: List[str] = kwargs.pop("contracts", [])
        self._ws_client_connection: Optional[WebSocketClient] = None
        self._subscription_required: bool = True
        self._missed_parts: bool = False
        self._last_subscription_check = None
        super().__init__(**kwargs)

    @property
    def params(self) -> Params:
        """Return params model."""

        return cast(Params, self.context.params)

    @property
    def subscription_status(self) -> SubscriptionStatus:
        """Returns subscription status"""
        return (
            self.context.shared_state.get(WEBSOCKET_SUBSCRIPTION_STATUS, {})
            .get(self.params.subscription_id, SubscriptionStatus.UNSUBSCRIBED)
            .value
        )

    @property
    def subscription_data(self) -> List[str]:
        """Returns subscription status"""
        return self.context.shared_state.get(WEBSOCKET_SUBSCRIPTIONS, {}).get(
            self.params.subscription_id, []
        )

    @property
    def subscribed(self) -> bool:
        return (
            SubscriptionStatus(self.subscription_status)
            == SubscriptionStatus.SUBSCRIBED
        )

    @property
    def subscribing(self) -> bool:
        return (
            SubscriptionStatus(self.subscription_status)
            == SubscriptionStatus.SUBSCRIBING
        )

    @property
    def checking_subscription(self) -> bool:
        return (
            SubscriptionStatus(self.subscription_status)
            == SubscriptionStatus.CHECKING_SUBSCRIPTION
        )

    @property
    def unsubscribed(self) -> bool:
        return (
            SubscriptionStatus(self.subscription_status)
            == SubscriptionStatus.UNSUBSCRIBED
        )

    @property
    def last_subscription_check(self) -> float:
        """Return when last subscription was checked."""
        if self._last_subscription_check is None:
            self._last_subscription_check = datetime.now().timestamp()
        return self._last_subscription_check

    def create_contract_subscription_payload(self, *args: Any, **kwargs: Any) -> str:
        """Create subscription payload."""

        raise NotImplementedError()

    def check_subscription(self) -> None:
        """Check for subscription status"""
        if datetime.now().timestamp() < self.last_subscription_check + 5:
            return
        self._check_subscription(subscription_id=self.params.subscription_id)
        self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
            self.params.subscription_id
        ] = SubscriptionStatus.CHECKING_SUBSCRIPTION
        self._last_subscription_check = datetime.now().timestamp()

    def act(self) -> None:
        """Perform subcription."""

        if self.subscribing or self.checking_subscription:
            return

        if self.subscribed:
            self.check_subscription()
            return

        if self.unsubscribed:
            self._create_subscription(
                provider=self.params.websocket_provider,
                subscription_id=self.params.subscription_id,
                subscription_payload=self.create_contract_subscription_payload(),
            )
            self.context.shared_state[WEBSOCKET_SUBSCRIPTION_STATUS][
                self.params.subscription_id
            ] = SubscriptionStatus.SUBSCRIBING
            return

    def _create_subscription(
        self,
        provider: str,
        subscription_id: str,
        subscription_payload: Optional[str] = None,
    ) -> Generator[None, None, WebsocketClientMessage]:
        """Subscribe to a websocket using websocket client connection."""
        self.context.logger.info(
            f"Creating websocket subscription using provider={provider} payload={subscription_payload}"
        )
        websocket_client_dialogues = cast(
            WebsocketClientDialogues, self.context.websocket_client_dialogues
        )
        (websocket_client_message, _) = websocket_client_dialogues.create(
            counterparty=str(WEBSOCKET_CLIENT_CONNECTION),
            performative=WebsocketClientMessage.Performative.SUBSCRIBE,
            subscription_id=subscription_id,
            url=provider,
            subscription_payload=subscription_payload,
        )
        self.context.outbox.put_message(
            message=websocket_client_message,
        )

    def _check_subscription(
        self,
        subscription_id: int,
    ) -> Generator[None, None, WebsocketClientMessage]:
        """Subscribe to a websocket using websocket client connection."""
        websocket_client_dialogues = cast(
            WebsocketClientDialogues, self.context.websocket_client_dialogues
        )
        (websocket_client_message, _) = websocket_client_dialogues.create(
            counterparty=str(WEBSOCKET_CLIENT_CONNECTION),
            performative=WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION,
            subscription_id=subscription_id,
        )
        self.context.outbox.put_message(
            message=websocket_client_message,
        )

    def _ws_send(
        self,
        payload: str,
        subscription_id: int,
    ) -> Generator[None, None, WebsocketClientMessage]:
        """Subscribe to a websocket using websocket client connection."""
        websocket_client_dialogues = cast(
            WebsocketClientDialogues, self.context.websocket_client_dialogues
        )
        (websocket_client_message, _) = websocket_client_dialogues.create(
            counterparty=str(WEBSOCKET_CLIENT_CONNECTION),
            performative=WebsocketClientMessage.Performative.SEND,
            payload=payload,
            subscription_id=subscription_id,
        )
        self.context.outbox.put_message(
            message=websocket_client_message,
        )
