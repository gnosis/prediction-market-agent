import typer

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler


def main(
    session_id: AgentIdentifier,
    delete_memories: bool = True,
    delete_prompts: bool = True,
    delete_goals: bool = True,
) -> None:
    """
    Delete all memories and prompts for a given agent, defined by the session_id.
    """
    if delete_prompts:
        prompt_handler = PromptTableHandler.from_agent_identifier(session_id)
        prompt_handler.delete_all_prompts()
        if prompt_handler.fetch_latest_prompt() is not None:
            raise Exception("Prompt entries were not deleted.")
        else:
            print("Prompt entries successfully deleted.")

    if delete_memories:
        long_term_memory = LongTermMemoryTableHandler.from_agent_identifier(session_id)
        long_term_memory.delete_all_memories()
        if len(long_term_memory.search()) != 0:
            raise Exception("Memory entries were not deleted.")
        else:
            print("Memory entries successfully deleted.")

    if delete_goals:
        evaluated_goal_table_handler = EvaluatedGoalTableHandler(agent_id=session_id)
        evaluated_goal_table_handler.delete_all_evaluated_goals()
        if len(evaluated_goal_table_handler.get_latest_evaluated_goals(limit=1)) != 0:
            raise Exception("Evaluated goal entries were not deleted.")
        else:
            print("Evaluated goal entries successfully deleted.")


if __name__ == "__main__":
    typer.run(main)
