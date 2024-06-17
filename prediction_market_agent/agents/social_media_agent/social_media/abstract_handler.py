import typing as t
from abc import ABCMeta, abstractmethod


class AbstractSocialMediaHandler(metaclass=ABCMeta):
    client: t.Any

    @abstractmethod
    def post(self, text: str, reasoning_reply_tweet: str) -> None:
        pass
