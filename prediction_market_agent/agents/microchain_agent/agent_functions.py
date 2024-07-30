import typing as t

from microchain import Agent, Function

from prediction_market_agent.agents.microchain_agent.prompts import (
    build_full_unformatted_system_prompt,
    extract_updatable_system_prompt,
)


class AgentAction(Function):
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        super().__init__()


class GetMyCurrentSystemPrompt(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to get your current system prompt."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return extract_updatable_system_prompt(self.agent.system_prompt)


class UpdateMySystemPrompt(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to update your system prompt."

    @property
    def example_args(self) -> list[str]:
        return ["This will be my new prompt."]

    def __call__(self, new_prompt: str) -> str:
        self.agent.system_prompt = build_full_unformatted_system_prompt(
            new_prompt
        ).format(engine_help=self.agent.engine.help)
        # History needs to be updated manually, because it's constructed only once in the agent initialization.
        self.agent.history[0] = dict(role="system", content=self.agent.system_prompt)
        return "The prompt has been updated"


AGENT_FUNCTIONS: list[t.Type[AgentAction]] = [
    GetMyCurrentSystemPrompt,
    UpdateMySystemPrompt,
]
