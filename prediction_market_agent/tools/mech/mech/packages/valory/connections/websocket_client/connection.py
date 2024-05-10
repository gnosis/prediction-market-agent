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

"""Websocket client connection."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, cast

import websocket
from aea.configurations.base import PublicId
from aea.connections.base import Connection, ConnectionStates
from aea.mail.base import Envelope
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue

from packages.valory.protocols.websocket_client.dialogues import WebsocketClientDialogue
from packages.valory.protocols.websocket_client.dialogues import (
    WebsocketClientDialogues as BaseWebsocketClientDialogues,
)
from packages.valory.protocols.websocket_client.message import WebsocketClientMessage


PUBLIC_ID = PublicId.from_str("valory/websocket_client:0.1.0")

DEFAULT_MAX_RETRIES = 5


class WebsocketClientDialogues(BaseWebsocketClientDialogues):
    """A class to keep track of IPFS dialogues."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize dialogues.

        :param kwargs: keyword arguments
        """

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> Dialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message

            :param message: an incoming/outgoing first message
            :param receiver_address: the address of the receiving agent
            :return: The role of the agent
            """
            return WebsocketClientDialogue.Role.CONNECTION

        BaseWebsocketClientDialogues.__init__(
            self,
            self_address=str(kwargs.pop("connection_id")),
            role_from_first_message=role_from_first_message,
            **kwargs,
        )


class WebsocketSubcription:
    """Websocket subscription."""

    _wss: websocket.WebSocket
    _outbox: Optional[asyncio.Queue]
    _envelope: Envelope
    _dialogue: WebsocketClientDialogue

    def __init__(
        self,
        subscription_id: str,
        outbox: asyncio.Queue,
        to: str,
        sender: str,
        logger: Optional[logging.Logger] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """Create a websocket subscription."""

        self._id = subscription_id
        self._url = None
        self._status = ConnectionStates.disconnected

        self._to = to
        self._sender = sender

        self._outbox = outbox
        self._executor = executor or ThreadPoolExecutor()
        self._loop = loop or asyncio.get_running_loop()

        self.logger = logger or logging.getLogger()

    @property
    def id(self) -> str:
        """Websocket id."""
        return self._id

    @property
    def url(self) -> str:
        """Returns the URL"""
        if self._url is None:
            raise ValueError(f"URL not set for websocket subscription {self.id}")
        return self._url

    @property
    def status(self) -> ConnectionStates:
        """Current status of the subscription."""
        return self._status

    def send(self, payload: str) -> int:
        """Send and return send length."""

        try:
            return self._wss.send(payload=payload)
        except websocket.WebSocketConnectionClosedException:
            self._status = ConnectionStates.disconnected
            return -1

    async def recv(self) -> None:
        """Run recv loop."""
        while self.status == ConnectionStates.connected:
            try:
                data = await self._loop.run_in_executor(
                    self._executor,
                    self._wss.recv,
                )
                message = WebsocketClientMessage(
                    performative=WebsocketClientMessage.Performative.RECV,
                    subscription_id=self.id,
                    data=data,
                )
            except (websocket.WebSocketConnectionClosedException, OSError) as e:
                self._status = ConnectionStates.disconnected
                message = WebsocketClientMessage(
                    performative=WebsocketClientMessage.Performative.ERROR,
                    message=f"Websocket connection disconnected with error {e}",
                    subscription_id=self.id,
                    alive=False,
                )

            await self._outbox.put(
                Envelope(
                    to=self._to,  # self._envelope.sender,
                    sender=self._sender,  # self._envelope.to,
                    message=message,
                )
            )

    async def subscribe(self, url: str) -> "WebsocketSubcription":
        """Subscribe to websocket."""
        # TODO: make these configurable
        tries = 0
        sleep = 3
        while tries < DEFAULT_MAX_RETRIES:
            try:
                self._status = ConnectionStates.connecting
                self._url = url
                self._wss = await self._loop.run_in_executor(
                    self._executor,
                    websocket.create_connection,
                    self.url,
                )
                self._status = ConnectionStates.connected
                return self
            except Exception as exception:  # pylint: disable=W0718
                tries += 1
                self.logger.error(
                    f"Failed to establish WebSocket connection: {exception}; "
                    f"URL: {self.url}; Try: {tries}; Will retry in {sleep} ..."
                )
                await asyncio.sleep(sleep)

        self._status = ConnectionStates.disconnected
        return self

    async def unsubscribe(self, payload: str = "") -> "WebsocketSubcription":
        """Unsubscribe from websocket."""
        self._status = ConnectionStates.disconnecting
        await self._loop.run_in_executor(
            self._executor,
            self._wss.close,
            websocket.STATUS_NORMAL,
            payload.encode(),
        )
        self._status = ConnectionStates.disconnected


class SubscriptionManager:
    """Websocket subscription manager."""

    _subscriptions: Dict[str, WebsocketSubcription]

    def __init__(
        self,
        outbox: asyncio.Queue,
        logger: Optional[logging.Logger] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """Websocket subscription manager."""

        self._subscriptions = {}
        self._executor = executor or ThreadPoolExecutor()
        self._loop = loop or asyncio.get_running_loop()
        self._outbox = outbox

        self.logger = logger or logging.getLogger()

    @property
    def outbox(self) -> asyncio.Queue:
        """Outbox."""
        return self._outbox

    def get(self, subscription_id: str) -> Optional[WebsocketSubcription]:
        """Returns a subscription by id"""
        return self._subscriptions.get(
            subscription_id,
        )

    async def create_subscription(
        self,
        url: str,
        subscription_id: str,
        to: str,
        sender: str,
    ) -> "WebsocketSubcription":
        """Create a websocket subscription."""
        self._subscriptions[subscription_id] = await WebsocketSubcription(
            subscription_id=subscription_id,
            to=to,
            sender=sender,
            outbox=self._outbox,
            logger=self.logger,
            loop=self._loop,
            executor=self._executor,
        ).subscribe(
            url=url,
        )
        return self._subscriptions[subscription_id]

    async def remove_subscription(
        self,
        subscription_id: int,
        payload: str = "",
    ) -> None:
        """Create a websocket subscription."""
        subscription = self._subscriptions.pop(subscription_id, None)
        if subscription is None:
            return
        await subscription.unsubscribe(payload=payload)

    async def remove_all_subscriptions(self) -> None:
        """Remove all subscriptions"""
        for sid, wss in self._subscriptions.items():
            self.logger.info(f"Unsubscribing from {sid}")
            await wss.unsubscribe()


class WebSocketClient(Connection):  # pylint: disable=Too many instance attributes
    """Proxy to the functionality of the SDK or API."""

    connection_id = PUBLIC_ID

    _executor: ThreadPoolExecutor
    _manager: SubscriptionManager
    _outbox: asyncio.Queue

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the connection."""
        super().__init__(**kwargs)

        self.dialogues = WebsocketClientDialogues(connection_id=PUBLIC_ID)

    @property
    def manager(self) -> SubscriptionManager:
        """Subscription manager."""
        return self._manager

    async def connect(self) -> None:
        """
        Set up the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """

        self._executor = ThreadPoolExecutor()
        self._outbox = asyncio.Queue()
        self._manager = SubscriptionManager(
            outbox=self._outbox,
            logger=self.logger,
            loop=self.loop,
            executor=self._executor,
        )

        self.state = ConnectionStates.connected
        self.logger.info("Websocket client established.")

    async def disconnect(self) -> None:
        """
        Tear down the connection.

        In the implementation, remember to update 'connection_status' accordingly.
        """

        await self._manager.remove_all_subscriptions()
        self._outbox.empty()
        self.state = ConnectionStates.disconnected

    async def send(self, envelope: Envelope) -> None:
        """
        Send an envelope.

        :param envelope: the envelope to send.
        """

        message = cast(WebsocketClientMessage, envelope.message)
        dialogue = self.dialogues.update(message)
        if message.performative == WebsocketClientMessage.Performative.SUBSCRIBE:
            response = await self.ws_subscribe(message=message, dialogue=dialogue)
        elif (
            message.performative
            == WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION
        ):
            response = self.ws_check_subscription(message=message, dialogue=dialogue)
        elif message.performative == WebsocketClientMessage.Performative.SEND:
            response = self.ws_send(message=message, dialogue=dialogue)
        else:
            raise ValueError(f"Invalid performative {message.performative}")

        await self._outbox.put(
            Envelope(
                to=envelope.sender,
                sender=envelope.to,
                message=response,
                context=envelope.context,
            )
        )

    async def ws_subscribe(
        self,
        message: WebsocketClientMessage,
        dialogue: WebsocketClientDialogue,
    ) -> Envelope:
        """Subscribe to a websocket."""
        wss = await self.manager.create_subscription(
            url=message.url,
            subscription_id=message.subscription_id,
            to=message.sender,
            sender=message.to,
        )
        if wss.status != ConnectionStates.connected:
            return self.error_message(
                message=message,
                dialogue=dialogue,
                error=f"Error subscribing to the websocket with id {message.subscription_id}",
            )

        self.loop.create_task(wss.recv())
        if message.subscription_payload is not None:
            wss.send(payload=message.subscription_payload)

        return cast(
            WebsocketClientMessage,
            dialogue.reply(
                performative=WebsocketClientMessage.Performative.SUBSCRIPTION,
                target_message=message,
                subscription_id=wss.id,
                alive=wss.status == ConnectionStates.connected,
            ),
        )

    def ws_check_subscription(
        self,
        message: WebsocketClientMessage,
        dialogue: WebsocketClientDialogue,
    ) -> Envelope:
        """Check websocket subscription."""
        wss = self.manager.get(subscription_id=message.subscription_id)
        if wss is None:
            return self.subscription_not_found_message(
                message=message, dialogue=dialogue
            )

        return cast(
            WebsocketClientMessage,
            dialogue.reply(
                performative=WebsocketClientMessage.Performative.SUBSCRIPTION,
                target_message=message,
                subscription_id=wss.id,
                alive=wss.status == ConnectionStates.connected,
            ),
        )

    def ws_send(
        self,
        message: WebsocketClientMessage,
        dialogue: WebsocketClientDialogue,
    ) -> Envelope:
        """Send data to subscription."""
        wss = self.manager.get(subscription_id=message.subscription_id)
        if wss is None:
            return self.subscription_not_found_message(
                message=message, dialogue=dialogue
            )

        send_length = wss.send(payload=message.payload)
        if send_length > -1:
            return cast(
                WebsocketClientMessage,
                dialogue.reply(
                    performative=WebsocketClientMessage.Performative.SEND_SUCCESS,
                    target_message=message,
                    subscription_id=message.subscription_id,
                    send_length=send_length,
                ),
            )

        return self.error_message(
            message=message,
            dialogue=dialogue,
            error=f"Error sending message; payload={message.payload}; subscription_id={message.subscription_id}",
            alive=wss.status == ConnectionStates.connected,
        )

    def subscription_not_found_message(
        self,
        message: WebsocketClientMessage,
        dialogue: WebsocketClientDialogue,
    ) -> WebsocketClientMessage:
        """Generate subscription not found message."""
        return cast(
            WebsocketClientMessage,
            dialogue.reply(
                performative=WebsocketClientMessage.Performative.ERROR,
                target_message=message,
                message=f"Subscription with ID {message.subscription_id} does not exist",
                subscription_id=message.subscription_id,
                alive=False,
            ),
        )

    def error_message(
        self,
        message: WebsocketClientMessage,
        dialogue: WebsocketClientDialogue,
        error: str,
        alive: bool = False,
    ) -> WebsocketClientMessage:
        """Generate error message."""
        return cast(
            WebsocketClientMessage,
            dialogue.reply(
                performative=WebsocketClientMessage.Performative.ERROR,
                target_message=message,
                subscription_id=message.subscription_id,
                message=error,
                alive=alive,
            ),
        )

    async def receive(self, *args: Any, **kwargs: Any) -> Envelope:
        """
        Receive an envelope. Blocking.

        :param args: arguments to receive
        :param kwargs: keyword arguments to receive
        :return: the envelope received, if present.  # noqa: DAR202
        """

        return await cast(Envelope, self._outbox.get())
