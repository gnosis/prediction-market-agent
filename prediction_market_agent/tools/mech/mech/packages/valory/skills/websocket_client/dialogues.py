"""
Dialogues
"""

from typing import Any

from aea.common import Address
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue as BaseDialogue
from aea.skills.base import Model

from packages.valory.protocols.websocket_client.dialogues import (
    WebsocketClientDialogue as BaseWebsocketClientDialogue,
)
from packages.valory.protocols.websocket_client.dialogues import (
    WebsocketClientDialogues as BaseWebsocketClientDialogues,
)

WebsocketClientDialogue = BaseWebsocketClientDialogue


class WebsocketClientDialogues(Model, BaseWebsocketClientDialogues):
    """The dialogues class keeps track of all dialogues."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize dialogues.

        :param kwargs: keyword arguments
        """
        Model.__init__(self, **kwargs)

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> BaseDialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message

            :param message: an incoming/outgoing first message
            :param receiver_address: the address of the receiving agent
            :return: The role of the agent
            """
            return WebsocketClientDialogue.Role.SKILL

        BaseWebsocketClientDialogues.__init__(
            self,
            self_address=str(self.skill_id),
            role_from_first_message=role_from_first_message,
        )
