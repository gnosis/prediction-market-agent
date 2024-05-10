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
from typing import Any, cast

from packages.valory.connections.websocket_client.connection import WebSocketClient
from packages.valory.skills.contract_subscription.handlers import DISCONNECTION_POINT
from packages.valory.skills.contract_subscription.models import Params
from packages.valory.skills.websocket_client.behaviours import (
    SubscriptionBehaviour as BaseSubscriptionBehaviour,
)
from packages.valory.skills.websocket_client.handlers import (
    SubscriptionStatus,
    WEBSOCKET_SUBSCRIPTION_STATUS,
)


DEFAULT_ENCODING = "utf-8"
WEBSOCKET_CLIENT_CONNECTION_NAME = "websocket_client"


class ContractSubscriptionBehaviour(BaseSubscriptionBehaviour):
    """This class scaffolds a behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the agent."""
        super().__init__(**kwargs)

    @property
    def params(self) -> Params:
        """Return params model."""

        return cast(Params, self.context.params)

    def setup(self) -> None:
        """Implement the setup."""
        self._last_subscription_check = None

        # if we are using polling, then we don't set up an contract subscription
        if self.params.use_polling:
            return

        for connection in self.context.outbox._multiplexer.connections:
            if connection.component_id.name == WEBSOCKET_CLIENT_CONNECTION_NAME:
                self._ws_client_connection = cast(WebSocketClient, connection)

    def create_contract_subscription_payload(self) -> str:
        """Create subscription payload."""
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["logs", {"address": self.params.contract_address}],
            }
        )

    def create_contract_filter_payload(self, disconnection_point: int) -> str:
        """Create subscription payload."""
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_newFilter",
                "params": [
                    {
                        "fromBlock": disconnection_point,
                        "address": self.params.contract_address,
                    }
                ],
            }
        )

    def act(self) -> None:
        """Perform subcription."""

        if self.params.use_polling:
            # do nothing if we are polling
            return

        if self.subscribing or self.checking_subscription:
            return

        disconnection_point = self.context.shared_state.get(DISCONNECTION_POINT, None)
        if self.subscribed and disconnection_point is not None:
            self.context.logger.info(
                f"Requesting block filter for {disconnection_point}"
            )
            self._ws_send(
                payload=self.create_contract_filter_payload(
                    disconnection_point=disconnection_point
                ),
                subscription_id=self.params.subscription_id,
            )
            self.context.shared_state[DISCONNECTION_POINT] = None

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
