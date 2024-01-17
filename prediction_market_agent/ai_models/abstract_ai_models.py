from typing import Optional


class AbstractAiChatModel:
    def complete(self, messages: list[str]) -> Optional[str]:
        """
        Execute the model, and return the completion result as a string.
        """
        raise NotImplementedError
