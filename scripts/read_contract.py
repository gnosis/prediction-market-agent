import os

from crewai import Agent, Crew, Task
from crewai_tools.tools.code_interpreter_tool.code_interpreter_tool import (
    CodeInterpreterTool,
)
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from langchain_openai import ChatOpenAI

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    ThinkThoroughlyBase,
)
from prediction_market_agent.utils import APIKeys

model = "gpt-4o-mini"

if __name__ == "__main__":
    llm = ChatOpenAI(
        model=model,
        api_key=APIKeys().openai_api_key.get_secret_value(),
        temperature=0.0,
    )
    os.environ["OPENAI_API_KEY"] = APIKeys().openai_api_key.get_secret_value()
    os.environ["SERPER_API_KEY"] = APIKeys().serper_api_key.get_secret_value()
    # Create an agent with code execution enabled
    serper_tool = SerperDevTool()
    # ToDo - Create tool for fetching ABI.
    coding_agent = Agent(
        role="Python Data Analyst",
        goal="Analyze data and provide insights using Python",
        backstory="You are an experienced data analyst with strong Python skills. You are also able to search the internet for contract's ABIs.",
        allow_code_execution=True,
        tools=[CodeInterpreterTool(user_dockerfile_path="Dockerfile"), serper_tool],
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    researcher = ThinkThoroughlyBase._get_researcher(model)

    # Create a task that requires code execution
    # data_analysis_task = Task(
    #     description="Install the numpy library and calculate the average age the array [11,22,33].",
    #     agent=coding_agent,
    #     expected_output="Average age of participants",
    # )
    research_task = Task(
        description="Find out the USDC token address on Gnosis Chain.",
        agent=researcher,
        expected_output="The token address",
    )
    coding_task = Task(
        description="Use the web3.py Python library and interact with the smart contract of token USDC on the Gnosis Chain in order to read the balance of wallet address 0xed56f76e9cbc6a64b821e9c016eafbd3db5436d1. Return the balance fetched using the latest block.",
        agent=coding_agent,
        expected_output="USDC balance of wallet address 0xed56f76e9cbc6a64b821e9c016eafbd3db5436d1",
    )
    # ToDo
    #  1. Find out USDC address on Gnosis using Tavily as tool
    #  2. Install web3 package
    #  3. Call the smart contract and find out balance of address

    # Create a crew and add the task
    analysis_crew = Crew(
        agents=[researcher, coding_agent], tasks=[research_task, coding_task]
    )

    # Execute the crew
    result = analysis_crew.kickoff()

    print(result)
