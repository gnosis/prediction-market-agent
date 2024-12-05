from microchain import Function


class SendTweet(Function):
    @property
    def description(self) -> str:
        return "Use this function to post a tweet on Twitter."

    @property
    def example_args(self) -> list[str]:
        return ["This is my message."]

    def __call__(
        self,
        message: str,
    ) -> str:
        # TODO: Complete the logic.
        return "Message sent."


TWITTER_FUNCTIONS: list[type[Function]] = [
    SendTweet,
]
