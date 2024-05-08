from abc import ABCMeta, abstractmethod
import typing as t


class AbstractSocialMediaHandler(metaclass=ABCMeta):
    client: t.Any

    @abstractmethod
    def post(self, text: str):
        pass
