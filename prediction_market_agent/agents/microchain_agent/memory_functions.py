from datetime import timedelta

from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from microchain import Function
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow

from prediction_market_agent.agents.microchain_agent.memory import DatedChatMessage
from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.utils import memories_to_learnings
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemories,
    LongTermMemoryTableHandler,
)


class LongTermMemoryBasedFunction(Function):
    def __init__(
        self, long_term_memory: LongTermMemoryTableHandler, model: str
    ) -> None:
        self.long_term_memory = long_term_memory
        self.model = model
        super().__init__()


class LookAtPastActionsFromLastDay(LongTermMemoryBasedFunction):
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
            DatedChatMessage.from_long_term_memory(ltm) for ltm in memories
        ]
        return memories_to_learnings(memories=simple_memories, model=self.model)


class CheckAllPastActionsGivenContext(LongTermMemoryBasedFunction):
    @property
    def description(self) -> str:
        return (
            "Use this function to fetch information about the actions you executed with respect to a specific context. "
            "For example, you can use this function to look into all your past actions if you ever did form a coalition with another agent."
        )

    @property
    def example_args(self) -> list[str]:
        return ["What coalitions did I form?"]

    def __call__(self, context: str) -> str:
        keys = MicrochainAgentKeys()
        all_memories = self.long_term_memory.search()

        collection = Chroma(
            embedding_function=OpenAIEmbeddings(
                api_key=keys.openai_api_key_secretstr_v1
            )
        )
        collection.add_texts(
            texts=[
                f"From: {check_not_none(x.metadata_dict)['role']} Content: {check_not_none(x.metadata_dict)['content']}"
                for x in all_memories
            ],
            metadatas=[{"json": x.model_dump_json()} for x in all_memories],
        )

        top_k_per_query_results = collection.similarity_search(context, k=50)
        results = [
            DatedChatMessage.from_long_term_memory(
                LongTermMemories.model_validate_json(x.metadata["json"])
            )
            for x in top_k_per_query_results
        ]

        return memories_to_learnings(memories=results, model=self.model)


MEMORY_FUNCTIONS: list[type[LongTermMemoryBasedFunction]] = [
    LookAtPastActionsFromLastDay,
    CheckAllPastActionsGivenContext,
]
