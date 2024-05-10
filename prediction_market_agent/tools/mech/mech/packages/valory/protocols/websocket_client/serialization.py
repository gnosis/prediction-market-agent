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

"""Serialization module for websocket_client protocol."""

# pylint: disable=too-many-statements,too-many-locals,no-member,too-few-public-methods,redefined-builtin
from typing import Any, Dict, cast

from aea.mail.base_pb2 import DialogueMessage
from aea.mail.base_pb2 import Message as ProtobufMessage
from aea.protocols.base import Message, Serializer

from packages.valory.protocols.websocket_client import websocket_client_pb2
from packages.valory.protocols.websocket_client.message import WebsocketClientMessage


class WebsocketClientSerializer(Serializer):
    """Serialization for the 'websocket_client' protocol."""

    @staticmethod
    def encode(msg: Message) -> bytes:
        """
        Encode a 'WebsocketClient' message into bytes.

        :param msg: the message object.
        :return: the bytes.
        """
        msg = cast(WebsocketClientMessage, msg)
        message_pb = ProtobufMessage()
        dialogue_message_pb = DialogueMessage()
        websocket_client_msg = websocket_client_pb2.WebsocketClientMessage()

        dialogue_message_pb.message_id = msg.message_id
        dialogue_reference = msg.dialogue_reference
        dialogue_message_pb.dialogue_starter_reference = dialogue_reference[0]
        dialogue_message_pb.dialogue_responder_reference = dialogue_reference[1]
        dialogue_message_pb.target = msg.target

        performative_id = msg.performative
        if performative_id == WebsocketClientMessage.Performative.SUBSCRIBE:
            performative = websocket_client_pb2.WebsocketClientMessage.Subscribe_Performative()  # type: ignore
            url = msg.url
            performative.url = url
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            if msg.is_set("subscription_payload"):
                performative.subscription_payload_is_set = True
                subscription_payload = msg.subscription_payload
                performative.subscription_payload = subscription_payload
            websocket_client_msg.subscribe.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.SUBSCRIPTION:
            performative = websocket_client_pb2.WebsocketClientMessage.Subscription_Performative()  # type: ignore
            alive = msg.alive
            performative.alive = alive
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.subscription.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION:
            performative = websocket_client_pb2.WebsocketClientMessage.Check_Subscription_Performative()  # type: ignore
            alive = msg.alive
            performative.alive = alive
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.check_subscription.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.SEND:
            performative = websocket_client_pb2.WebsocketClientMessage.Send_Performative()  # type: ignore
            payload = msg.payload
            performative.payload = payload
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.send.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.SEND_SUCCESS:
            performative = websocket_client_pb2.WebsocketClientMessage.Send_Success_Performative()  # type: ignore
            send_length = msg.send_length
            performative.send_length = send_length
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.send_success.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.RECV:
            performative = websocket_client_pb2.WebsocketClientMessage.Recv_Performative()  # type: ignore
            data = msg.data
            performative.data = data
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.recv.CopyFrom(performative)
        elif performative_id == WebsocketClientMessage.Performative.ERROR:
            performative = websocket_client_pb2.WebsocketClientMessage.Error_Performative()  # type: ignore
            alive = msg.alive
            performative.alive = alive
            message = msg.message
            performative.message = message
            subscription_id = msg.subscription_id
            performative.subscription_id = subscription_id
            websocket_client_msg.error.CopyFrom(performative)
        else:
            raise ValueError("Performative not valid: {}".format(performative_id))

        dialogue_message_pb.content = websocket_client_msg.SerializeToString()

        message_pb.dialogue_message.CopyFrom(dialogue_message_pb)
        message_bytes = message_pb.SerializeToString()
        return message_bytes

    @staticmethod
    def decode(obj: bytes) -> Message:
        """
        Decode bytes into a 'WebsocketClient' message.

        :param obj: the bytes object.
        :return: the 'WebsocketClient' message.
        """
        message_pb = ProtobufMessage()
        websocket_client_pb = websocket_client_pb2.WebsocketClientMessage()
        message_pb.ParseFromString(obj)
        message_id = message_pb.dialogue_message.message_id
        dialogue_reference = (
            message_pb.dialogue_message.dialogue_starter_reference,
            message_pb.dialogue_message.dialogue_responder_reference,
        )
        target = message_pb.dialogue_message.target

        websocket_client_pb.ParseFromString(message_pb.dialogue_message.content)
        performative = websocket_client_pb.WhichOneof("performative")
        performative_id = WebsocketClientMessage.Performative(str(performative))
        performative_content = dict()  # type: Dict[str, Any]
        if performative_id == WebsocketClientMessage.Performative.SUBSCRIBE:
            url = websocket_client_pb.subscribe.url
            performative_content["url"] = url
            subscription_id = websocket_client_pb.subscribe.subscription_id
            performative_content["subscription_id"] = subscription_id
            if websocket_client_pb.subscribe.subscription_payload_is_set:
                subscription_payload = (
                    websocket_client_pb.subscribe.subscription_payload
                )
                performative_content["subscription_payload"] = subscription_payload
        elif performative_id == WebsocketClientMessage.Performative.SUBSCRIPTION:
            alive = websocket_client_pb.subscription.alive
            performative_content["alive"] = alive
            subscription_id = websocket_client_pb.subscription.subscription_id
            performative_content["subscription_id"] = subscription_id
        elif performative_id == WebsocketClientMessage.Performative.CHECK_SUBSCRIPTION:
            alive = websocket_client_pb.check_subscription.alive
            performative_content["alive"] = alive
            subscription_id = websocket_client_pb.check_subscription.subscription_id
            performative_content["subscription_id"] = subscription_id
        elif performative_id == WebsocketClientMessage.Performative.SEND:
            payload = websocket_client_pb.send.payload
            performative_content["payload"] = payload
            subscription_id = websocket_client_pb.send.subscription_id
            performative_content["subscription_id"] = subscription_id
        elif performative_id == WebsocketClientMessage.Performative.SEND_SUCCESS:
            send_length = websocket_client_pb.send_success.send_length
            performative_content["send_length"] = send_length
            subscription_id = websocket_client_pb.send_success.subscription_id
            performative_content["subscription_id"] = subscription_id
        elif performative_id == WebsocketClientMessage.Performative.RECV:
            data = websocket_client_pb.recv.data
            performative_content["data"] = data
            subscription_id = websocket_client_pb.recv.subscription_id
            performative_content["subscription_id"] = subscription_id
        elif performative_id == WebsocketClientMessage.Performative.ERROR:
            alive = websocket_client_pb.error.alive
            performative_content["alive"] = alive
            message = websocket_client_pb.error.message
            performative_content["message"] = message
            subscription_id = websocket_client_pb.error.subscription_id
            performative_content["subscription_id"] = subscription_id
        else:
            raise ValueError("Performative not valid: {}.".format(performative_id))

        return WebsocketClientMessage(
            message_id=message_id,
            dialogue_reference=dialogue_reference,
            target=target,
            performative=performative,
            **performative_content
        )
