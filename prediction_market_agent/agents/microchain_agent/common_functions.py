from microchain import Function
from prediction_market_agent_tooling.tools.utils import utcnow


class TodayDate(Function):
    @property
    def description(self) -> str:
        return "Use this function to get the current date."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        now = utcnow()
        return f"Today is {now.strftime('%Y-%m-%d %H:%M:%S')}. The day is {now.strftime('%A')}."


COMMON_FUNCIONS: list[type[Function]] = [
    TodayDate,
]
