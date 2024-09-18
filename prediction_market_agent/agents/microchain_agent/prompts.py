from enum import Enum

from pydantic import BaseModel

NON_UPDATABLE_DIVIDOR = "You can use the following functions:"  # Agent can update only part of the prompt before this string.
SYSTEM_PROMPT = f"""Act as an autonomous agent with a goal to learn, survive and evolve.

{NON_UPDATABLE_DIVIDOR}

{{engine_help}}

Only output valid Python function calls, without code formatting characters, without any other text. i.e. it should run if passed to Python's `eval` function.
Only output a single function call per message.
"""
# This is the original system prompt of the microchain agent, users on Streamlit can select this,
# instead of always waiting for the agent to learn trading.
TRADING_AGENT_SYSTEM_PROMPT = f"""Act as a trader agent in prediction markets to maximise your profit.

Research markets, buy tokens you consider undervalued, and sell tokens that you
hold and consider overvalued.

You need to reason step by step. Start by assessing your current positions and balance. 
Do you have any positions in the markets returned from GetMarkets? 
Consider selling overvalued tokens AND buying undervalued tokens.

You know everything needed and now just trade on the markets.

{NON_UPDATABLE_DIVIDOR}

{{engine_help}}

Only output valid Python function calls, without code formatting characters, without any other text. i.e. it should run if passed to Python's `eval` function.
Only output a single function call per message.
Make 'Reasoning' calls frequently - at least every other call.
"""

# This is similar to the TRADING_AGENT_SYSTEM_PROMPT, except that it doesn't
# contain any specific instructions on what to do. This is appropriate to use
# for an agent when combined with a user-prompt containing the instructions for
# the session.
TRADING_AGENT_SYSTEM_PROMPT_MINIMAL = f"""You are a helpful assistant, who specializes as an expert trader agent in prediction markets.

{NON_UPDATABLE_DIVIDOR}

{{engine_help}}

Only output valid Python function calls, without code formatting characters, without any other text. i.e. it should run if passed to Python's `eval` function.
Only output a single function call per message.
Make 'Reasoning' calls frequently - at least every other call. You need to reason step by step.
"""

# Experimental system prompt for job-solving agent.
JOB_AGENT_SYSTEM_PROMPT = f"""Act as a job-solving agents that picks up available jobs and completes them for getting rewards.

Pick up available job that's returned from GetJobs and pick one that you can complete and it's worth finishing.

While working on a job, reason step by step and write down thoroughly the process. You can use the 'Reasoning' function for that.

Don't do anything else, just complete the job, and then pick up another one.

{NON_UPDATABLE_DIVIDOR}

{{engine_help}}

Only output valid Python function calls, without code formatting characters, without any other text.
Only output a single function call per message.
Make 'Reasoning' calls frequently - at least every other call.
"""


def extract_updatable_system_prompt(system_prompt: str) -> str:
    if NON_UPDATABLE_DIVIDOR not in system_prompt:
        raise ValueError("The system prompt doesn't contain the non-updatable part.")

    return system_prompt.split(NON_UPDATABLE_DIVIDOR)[0]


def build_full_unformatted_system_prompt(system_prompt: str) -> str:
    if NON_UPDATABLE_DIVIDOR in system_prompt:
        raise ValueError("The system prompt already contains the non-updatable part.")

    return (
        system_prompt
        + "\n\n"
        + NON_UPDATABLE_DIVIDOR
        + SYSTEM_PROMPT.split(NON_UPDATABLE_DIVIDOR)[1]
    )


class SystemPromptChoice(str, Enum):
    JUST_BORN = "just_born"
    TRADING_AGENT = "trading_agent"
    TRADING_AGENT_MINIMAL = "trading_agent_minimal"
    JOB_AGENT = "job_agent"


class FunctionsConfig(BaseModel):
    # TODO: We need better logic here: https://github.com/gnosis/prediction-market-agent/issues/350
    include_learning_functions: bool
    include_trading_functions: bool
    include_universal_functions: bool
    include_agent_functions: bool

    @staticmethod
    def from_system_prompt_choice(
        system_prompt_choice: SystemPromptChoice,
    ) -> "FunctionsConfig":
        include_trading_functions = False
        include_learning_functions = False
        include_universal_functions = False
        include_agent_functions = False

        if system_prompt_choice == SystemPromptChoice.JUST_BORN:
            include_learning_functions = True
            include_agent_functions = True
            include_trading_functions = True

        elif system_prompt_choice in [
            SystemPromptChoice.TRADING_AGENT,
            SystemPromptChoice.TRADING_AGENT_MINIMAL,
        ]:
            include_trading_functions = True

        elif system_prompt_choice == SystemPromptChoice.JOB_AGENT:
            include_universal_functions = True
            include_agent_functions = True
            include_trading_functions = True

        return FunctionsConfig(
            include_trading_functions=include_trading_functions,
            include_learning_functions=include_learning_functions,
            include_universal_functions=include_universal_functions,
            include_agent_functions=include_agent_functions,
        )


SYSTEM_PROMPTS: dict[SystemPromptChoice, str] = {
    SystemPromptChoice.JUST_BORN: SYSTEM_PROMPT,
    SystemPromptChoice.TRADING_AGENT: TRADING_AGENT_SYSTEM_PROMPT,
    SystemPromptChoice.JOB_AGENT: JOB_AGENT_SYSTEM_PROMPT,
    SystemPromptChoice.TRADING_AGENT_MINIMAL: TRADING_AGENT_SYSTEM_PROMPT_MINIMAL,
}
