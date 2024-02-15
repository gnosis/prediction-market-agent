from typing import Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class AbstractAiChatModel:
    def complete(self, messages: list[Message]) -> Optional[str]:
        """
        Execute the model, and return the completion result as a string.
        """
        raise NotImplementedError
