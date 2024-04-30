import typer
from microchain import LLM, Agent, Engine, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from functions import MARKET_FUNCTIONS, MISC_FUNCTIONS, SummarizeLearnings
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.omen_functions import (
    OMEN_FUNCTIONS,
)
from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.utils import APIKeys


def main(
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4-turbo-preview",
) -> None:
    engine = Engine()
    engine.register(Reasoning())
    engine.register(Stop())
    for function in MISC_FUNCTIONS:
        engine.register(function())
    for function in MARKET_FUNCTIONS:
        engine.register(function(market_type=MarketType.OMEN))
    for function in OMEN_FUNCTIONS:
        engine.register(function())

    # This description below serves to unique identify agent entries on the LTM, and should be
    # unique across instances (i.e. markets).
    unique_task_description = f"microchain-agent-demo"
    long_term_memory = LongTermMemory(unique_task_description, DBStorage())
    engine.register(SummarizeLearnings(long_term_memory))

    generator = OpenAIChatGenerator(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        api_base=api_base,
        temperature=0.7,
    )
    agent = Agent(llm=LLM(generator=generator), engine=engine)
    agent.prompt = f"""Act as a agent to maximise your profit. You can use the following functions:
    
    {engine.help}
    
    Do not call other functions except for the GetMarket function and the GetBalance.
    
    Only output valid Python function calls.
    
    """

    agent.bootstrap = ['Reasoning("I need to reason step-by-step")']
    agent.run(iterations=2)
    # generator.print_usage() # Waiting for microchain release
    # ToDo - Add agent.history to the DB
    print("finished")


if __name__ == "__main__":
    typer.run(main)
