import typing as t

from microchain import Agent, Function

from prediction_market_agent.agents.microchain_agent.prompts import (
    NON_UPDATABLE_DIVIDOR,
    SYSTEM_PROMPT,
)


class AgentAction(Function):
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        super().__init__()


class GetMyCurrentPrompt(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to get your current prompt."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return str(self.agent.prompt).split(NON_UPDATABLE_DIVIDOR)[0].strip()


class UpdateMyPrompt(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to update your prompt."

    @property
    def example_args(self) -> list[str]:
        return ["This will be my new prompt."]

    def __call__(self, new_prompt: str) -> str:
        prompt_template = (
            new_prompt
            + "\n\n"
            + NON_UPDATABLE_DIVIDOR
            + SYSTEM_PROMPT.split(NON_UPDATABLE_DIVIDOR)[1]
        )
        self.agent.prompt = prompt_template.format(engine_help=self.agent.engine.help)
        # History needs to be updated manually, because it's constructed only once in the agent initialization.
        self.agent.history[0] = dict(role="system", content=self.agent.prompt)
        return "The prompt has been updated"


class GetMyCurrentBootstrap(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to get your current bootstrap."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return (
            str(self.agent.bootstrap[0])
            if self.agent.bootstrap
            else "No bootstrap available."
        )


class UpdateMyBootstrap(AgentAction):
    @property
    def description(self) -> str:
        return "Use this function to update your bootstrap."

    @property
    def example_args(self) -> list[str]:
        return ["This will be my new bootstrap."]

    def __call__(self, new_bootstrap: str) -> str:
        self.agent.bootstrap = [new_bootstrap]
        return "The bootstrap has been updated"


AGENT_FUNCTIONS: list[t.Type[AgentAction]] = [
    GetMyCurrentPrompt,
    UpdateMyPrompt,
    GetMyCurrentBootstrap,
    UpdateMyBootstrap,
]
