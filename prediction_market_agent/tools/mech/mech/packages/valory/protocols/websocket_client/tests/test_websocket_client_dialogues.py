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

"""Test dialogues module for websocket_client protocol."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,redefined-builtin
from aea.test_tools.test_protocol import BaseProtocolDialoguesTestCase

from packages.valory.protocols.websocket_client.dialogues import (
    WebsocketClientDialogue,
    WebsocketClientDialogues,
)
from packages.valory.protocols.websocket_client.message import WebsocketClientMessage


class TestDialoguesWebsocketClient(BaseProtocolDialoguesTestCase):
    """Test for the 'websocket_client' protocol dialogues."""

    MESSAGE_CLASS = WebsocketClientMessage

    DIALOGUE_CLASS = WebsocketClientDialogue

    DIALOGUES_CLASS = WebsocketClientDialogues

    ROLE_FOR_THE_FIRST_MESSAGE = WebsocketClientDialogue.Role.CONNECTION  # CHECK

    def make_message_content(self) -> dict:
        """Make a dict with message contruction content for dialogues.create."""
        return dict(
            performative=WebsocketClientMessage.Performative.SUBSCRIBE,
            url="some str",
            subscription_id="some str",
            subscription_payload="some str",
        )
