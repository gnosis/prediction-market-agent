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

"""This module contains websocket_client's message definition."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,too-many-branches,not-an-iterable,unidiomatic-typecheck,unsubscriptable-object
import logging
from typing import Any, Optional, Set, Tuple, cast

from aea.configurations.base import PublicId
from aea.exceptions import AEAEnforceError, enforce
from aea.protocols.base import Message


_default_logger = logging.getLogger(
    "aea.packages.valory.protocols.websocket_client.message"
)

DEFAULT_BODY_SIZE = 4


class WebsocketClientMessage(Message):
    """A protocol for websocket client."""

    protocol_id = PublicId.from_str("valory/websocket_client:0.1.0")
    protocol_specification_id = PublicId.from_str("valory/websocket_client:1.0.0")

    class Performative(Message.Performative):
        """Performatives for the websocket_client protocol."""

        CHECK_SUBSCRIPTION = "check_subscription"
        ERROR = "error"
        RECV = "recv"
        SEND = "send"
        SEND_SUCCESS = "send_success"
        SUBSCRIBE = "subscribe"
        SUBSCRIPTION = "subscription"

        def __str__(self) -> str:
            """Get the string representation."""
            return str(self.value)

    _performatives = {
        "check_subscription",
        "error",
        "recv",
        "send",
        "send_success",
        "subscribe",
        "subscription",
    }
    __slots__: Tuple[str, ...] = tuple()

    class _SlotsCls:
        __slots__ = (
            "alive",
            "data",
            "dialogue_reference",
            "message",
            "message_id",
            "payload",
            "performative",
            "send_length",
            "subscription_id",
            "subscription_payload",
            "target",
            "url",
        )

    def __init__(
        self,
        performative: Performative,
        dialogue_reference: Tuple[str, str] = ("", ""),
        message_id: int = 1,
        target: int = 0,
        **kwargs: Any,
    ):
        """
        Initialise an instance of WebsocketClientMessage.

        :param message_id: the message id.
        :param dialogue_reference: the dialogue reference.
        :param target: the message target.
        :param performative: the message performative.
        :param **kwargs: extra options.
        """
        super().__init__(
            dialogue_reference=dialogue_reference,
            message_id=message_id,
            target=target,
            performative=WebsocketClientMessage.Performative(performative),
            **kwargs,
        )

    @property
    def valid_performatives(self) -> Set[str]:
        """Get valid performatives."""
        return self._performatives

    @property
    def dialogue_reference(self) -> Tuple[str, str]:
        """Get the dialogue_reference of the message."""
        enforce(self.is_set("dialogue_reference"), "dialogue_reference is not set.")
        return cast(Tuple[str, str], self.get("dialogue_reference"))

    @property
    def message_id(self) -> int:
        """Get the message_id of the message."""
        enforce(self.is_set("message_id"), "message_id is not set.")
        return cast(int, self.get("message_id"))

    @property
    def performative(self) -> Performative:  # type: ignore # noqa: F821
        """Get the performative of the message."""
        enforce(self.is_set("performative"), "performative is not set.")
        return cast(WebsocketClientMessage.Performative, self.get("performative"))

    @property
    def target(self) -> int:
        """Get the target of the message."""
        enforce(self.is_set("target"), "target is not set.")
        return cast(int, self.get("target"))

    @property
    def alive(self) -> bool:
        """Get the 'alive' content from the message."""
        enforce(self.is_set("alive"), "'alive' content is not set.")
        return cast(bool, self.get("alive"))

    @property
    def data(self) -> str:
        """Get the 'data' content from the message."""
        enforce(self.is_set("data"), "'data' content is not set.")
        return cast(str, self.get("data"))

    @property
    def message(self) -> str:
        """Get the 'message' content from the message."""
        enforce(self.is_set("message"), "'message' content is not set.")
        return cast(str, self.get("message"))

    @property
    def payload(self) -> str:
        """Get the 'payload' content from the message."""
        enforce(self.is_set("payload"), "'payload' content is not set.")
        return cast(str, self.get("payload"))

    @property
    def send_length(self) -> int:
        """Get the 'send_length' content from the message."""
        enforce(self.is_set("send_length"), "'send_length' content is not set.")
        return cast(int, self.get("send_length"))

    @property
    def subscription_id(self) -> str:
        """Get the 'subscription_id' content from the message."""
        enforce(self.is_set("subscription_id"), "'subscription_id' content is not set.")
        return cast(str, self.get("subscription_id"))

    @property
    def subscription_payload(self) -> Optional[str]:
        """Get the 'subscription_payload' content from the message."""
        return cast(Optional[str], self.get("subscription_payload"))

    @property
    def url(self) -> str:
        """Get the 'url' content from the message."""
        enforce(self.is_set("url"), "'url' content is not set.")
        return cast(str, self.get("url"))

    def _is_consistent(self) -> bool:
        """Check that the message follows the websocket_client protocol."""
        try:
            enforce(
                isinstance(self.dialogue_reference, tuple),
                "Invalid type for 'dialogue_reference'. Expected 'tuple'. Found '{}'.".format(
                    type(self.dialogue_reference)
                ),
            )
            enforce(
                isinstance(self.dialogue_reference[0], str),
                "Invalid type for 'dialogue_reference[0]'. Expected 'str'. Found '{}'.".format(
                    type(self.dialogue_reference[0])
                ),
            )
            enforce(
                isinstance(self.dialogue_reference[1], str),
                "Invalid type for 'dialogue_reference[1]'. Expected 'str'. Found '{}'.".format(
                    type(self.dialogue_reference[1])
                ),
            )
            enforce(
                type(self.message_id) is int,
                "Invalid type for 'message_id'. Expected 'int'. Found '{}'.".format(
                    type(self.message_id)
                ),
            )
            enforce(
                type(self.target) is int,
                "Invalid type for 'target'. Expected 'int'. Found '{}'.".format(
                    type(self.target)
                ),
            )

            # Light Protocol Rule 2
            # Check correct performative
            enforce(
                isinstance(self.performative, WebsocketClientMessage.Performative),
                "Invalid 'performative'. Expected either of '{}'. Found '{}'.".format(
                    self.valid_performatives, self.performative
                ),
            )

            # Check correct contents
            actual_nb_of_contents = len(self._body) - DEFAULT_BODY_SIZE
            expected_nb_of_contents = 0
            if self.performative == WebsocketClientMessage.Performative.SUBSCRIBE:
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.url, str),
                    "Invalid type for content 'url'. Expected 'str'. Found '{}'.".format(
                        type(self.url)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
                if self.is_set("subscription_payload"):
                    expected_nb_of_contents += 1
                    subscription_payload = cast(str, self.subscription_payload)
                    enforce(
                        isinstance(subscription_payload, str),
                        "Invalid type for content 'subscription_payload'. Expected 'str'. Found '{}'.".format(
                            type(subscription_payload)
                        ),
                    )
            elif self.performative == WebsocketClientMessage.Performative.SUBSCRIPTION:
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.alive, bool),
                    "Invalid type for content 'alive'. Expected 'bool'. Found '{}'.".format(
                        type(self.alive)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
            elif (
                self.performative
                == WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION
            ):
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.alive, bool),
                    "Invalid type for content 'alive'. Expected 'bool'. Found '{}'.".format(
                        type(self.alive)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
            elif self.performative == WebsocketClientMessage.Performative.SEND:
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.payload, str),
                    "Invalid type for content 'payload'. Expected 'str'. Found '{}'.".format(
                        type(self.payload)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
            elif self.performative == WebsocketClientMessage.Performative.SEND_SUCCESS:
                expected_nb_of_contents = 2
                enforce(
                    type(self.send_length) is int,
                    "Invalid type for content 'send_length'. Expected 'int'. Found '{}'.".format(
                        type(self.send_length)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
            elif self.performative == WebsocketClientMessage.Performative.RECV:
                expected_nb_of_contents = 2
                enforce(
                    isinstance(self.data, str),
                    "Invalid type for content 'data'. Expected 'str'. Found '{}'.".format(
                        type(self.data)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )
            elif self.performative == WebsocketClientMessage.Performative.ERROR:
                expected_nb_of_contents = 3
                enforce(
                    isinstance(self.alive, bool),
                    "Invalid type for content 'alive'. Expected 'bool'. Found '{}'.".format(
                        type(self.alive)
                    ),
                )
                enforce(
                    isinstance(self.message, str),
                    "Invalid type for content 'message'. Expected 'str'. Found '{}'.".format(
                        type(self.message)
                    ),
                )
                enforce(
                    isinstance(self.subscription_id, str),
                    "Invalid type for content 'subscription_id'. Expected 'str'. Found '{}'.".format(
                        type(self.subscription_id)
                    ),
                )

            # Check correct content count
            enforce(
                expected_nb_of_contents == actual_nb_of_contents,
                "Incorrect number of contents. Expected {}. Found {}".format(
                    expected_nb_of_contents, actual_nb_of_contents
                ),
            )

            # Light Protocol Rule 3
            if self.message_id == 1:
                enforce(
                    self.target == 0,
                    "Invalid 'target'. Expected 0 (because 'message_id' is 1). Found {}.".format(
                        self.target
                    ),
                )
        except (AEAEnforceError, ValueError, KeyError) as e:
            _default_logger.error(str(e))
            return False

        return True
