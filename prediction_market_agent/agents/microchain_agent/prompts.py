from pydantic import BaseModel


class FunctionsConfig(BaseModel):
    common_functions: bool = False
    include_learning_functions: bool = False
    include_trading_functions: bool = False
    include_universal_functions: bool = False
    include_agent_functions: bool = False
    include_job_functions: bool = False
    include_sending_functions: bool = False
    include_twitter_functions: bool = False
    include_messages_functions: bool = False
    include_nft_functions: bool = False
    balance_functions: bool = False

    def combine(self, other: "FunctionsConfig") -> "FunctionsConfig":
        """
        Allow to combine two functions config into one, where the fields are OR-ed.
        """
        combined = {}
        for field in self.model_fields:
            combined[field] = getattr(self, field) or getattr(other, field)
        return FunctionsConfig(**combined)


class SystemPromptConfig(BaseModel):
    name: str
    system_prompt: str
    functions_config: FunctionsConfig


NON_UPDATABLE_DIVIDOR = "You can use the following functions:"  # Agent can update only part of the prompt before this string.
REQUIRED_PROMPT_ENDING_FOR_MICROCHAIN = f"""

{NON_UPDATABLE_DIVIDOR}

{{engine_help}}

Only output valid Python function calls, without code formatting characters, without any other text.
Only output a single function call per message.
Make 'Reasoning' calls frequently - at least every other call.
If some function errors out, include the exact function and arguments you tried to call in your follow-up reasoning call.
"""

JUST_BORN_SYSTEM_PROMPT_CONFIG = SystemPromptConfig(
    name="just born",
    system_prompt=f"""Act as an autonomous agent with a goal to learn, survive and evolve.
""",
    functions_config=FunctionsConfig(
        include_learning_functions=True,
        include_agent_functions=True,
        include_trading_functions=True,
        include_job_functions=True,
        include_universal_functions=True,
    ),
)

# This is the original system prompt of the microchain agent, users on Streamlit can select this,
# instead of always waiting for the agent to learn trading.
TRADING_AGENT_SYSTEM_PROMPT_CONFIG = SystemPromptConfig(
    name="trading",
    system_prompt=f"""Act as a trader agent in prediction markets to maximise your profit.

Research markets, buy tokens you consider undervalued, and sell tokens that you
hold and consider overvalued.

You need to reason step by step. Start by assessing your current positions and balance. 
Do you have any positions in the markets returned from GetMarkets? 
Consider selling overvalued tokens AND buying undervalued tokens.

You know everything needed and now just trade on the markets.
""",
    functions_config=FunctionsConfig(
        include_trading_functions=True,
    ),
)

# This is similar to the TRADING_AGENT_SYSTEM_PROMPT, except that it doesn't
# contain any specific instructions on what to do. This is appropriate to use
# for an agent when combined with a user-prompt containing the instructions for
# the session.
TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG = SystemPromptConfig(
    name="trading minimal",
    system_prompt=f"""You are a helpful assistant, who specializes as an expert trader agent in prediction markets.
""",
    functions_config=FunctionsConfig(
        include_trading_functions=True,
    ),
)

# Experimental system prompt for job-solving agent.
JOB_AGENT_SYSTEM_PROMPT_CONFIG = SystemPromptConfig(
    name="job solving",
    system_prompt=f"""Act as a job-solving agents that picks up available jobs and completes them for getting rewards.

Pick up available job that's returned from GetJobs and pick one that you can complete and it's worth finishing.

While working on a job, reason step by step and write down thoroughly the process. You can use the 'Reasoning' function for that.

Don't do anything else, just complete the job, and then pick up another one.
""",
    functions_config=FunctionsConfig(
        include_universal_functions=True,
        include_agent_functions=True,
        include_trading_functions=True,
        include_job_functions=True,
    ),
)


def extract_updatable_system_prompt(system_prompt: str) -> str:
    if NON_UPDATABLE_DIVIDOR not in system_prompt:
        raise ValueError("The system prompt doesn't contain the non-updatable part.")

    return system_prompt.split(NON_UPDATABLE_DIVIDOR)[0]


def build_full_unformatted_system_prompt(system_prompt: str) -> str:
    if NON_UPDATABLE_DIVIDOR in system_prompt:
        raise ValueError("The system prompt already contains the non-updatable part.")

    return system_prompt + REQUIRED_PROMPT_ENDING_FOR_MICROCHAIN
