from datetime import timedelta

from microchain import Function
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.microchain_agent.memory import (
    LongTermMemory,
    SimpleMemoryMicrochain,
)
from prediction_market_agent.agents.utils import memories_to_learnings


class RememberPastActions(Function):
    def __init__(self, long_term_memory: LongTermMemory, model: str) -> None:
        self.long_term_memory = long_term_memory
        self.model = model
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Use this function to fetch information about the actions you "
            "executed over the past day. Examples of past activities include "
            "previous bets you placed, previous markets you redeemed from, "
            "balances you requested, market positions you requested, markets "
            "you fetched, tokens you bought, tokens you sold, probabilities "
            "for markets you requested, among others."
        )

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        # Get the last day's of the agent's memory. Add a +1hour buffer to
        # make sure a cronjob-scheduled agent that calls this in the middle of
        # its run doesn't miss anything from the previous day.
        memories = self.long_term_memory.search(from_=utcnow() - timedelta(hours=25))
        simple_memories = [
            SimpleMemoryMicrochain.from_long_term_memory(ltm) for ltm in memories
        ]
        return memories_to_learnings(memories=simple_memories, model=self.model)
