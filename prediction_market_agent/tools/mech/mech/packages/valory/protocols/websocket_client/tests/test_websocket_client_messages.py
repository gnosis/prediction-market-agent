# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 valory
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

"""Test messages module for websocket_client protocol."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,redefined-builtin
from typing import List

from aea.test_tools.test_protocol import BaseProtocolMessagesTestCase

from packages.valory.protocols.websocket_client.message import WebsocketClientMessage


class TestMessageWebsocketClient(BaseProtocolMessagesTestCase):
    """Test for the 'websocket_client' protocol message."""

    MESSAGE_CLASS = WebsocketClientMessage

    def build_messages(self) -> List[WebsocketClientMessage]:  # type: ignore[override]
        """Build the messages to be used for testing."""
        return [
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SUBSCRIBE,
                url="some str",
                subscription_id="some str",
                subscription_payload="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SUBSCRIPTION,
                alive=True,
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION,
                alive=True,
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SEND,
                payload="some str",
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SEND_SUCCESS,
                send_length=12,
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.RECV,
                data="some str",
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.ERROR,
                alive=True,
                message="some str",
                subscription_id="some str",
            ),
        ]

    def build_inconsistent(self) -> List[WebsocketClientMessage]:  # type: ignore[override]
        """Build inconsistent messages to be used for testing."""
        return [
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SUBSCRIBE,
                # skip content: url
                subscription_id="some str",
                subscription_payload="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SUBSCRIPTION,
                # skip content: alive
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION,
                # skip content: alive
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SEND,
                # skip content: payload
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.SEND_SUCCESS,
                # skip content: send_length
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.RECV,
                # skip content: data
                subscription_id="some str",
            ),
            WebsocketClientMessage(
                performative=WebsocketClientMessage.Performative.ERROR,
                # skip content: alive
                message="some str",
                subscription_id="some str",
            ),
        ]
