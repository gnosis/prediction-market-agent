import os

from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI

from prediction_market_agent.utils import APIKeys

model = "gpt-4o-mini"

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = APIKeys().openai_api_key.get_secret_value()
    # Create an agent with code execution enabled
    coding_agent = Agent(
        role="Python Data Analyst",
        goal="Analyze data and provide insights using Python",
        backstory="You are an experienced data analyst with strong Python skills.",
        allow_code_execution=True,
        llm=ChatOpenAI(
            model=model,
            api_key=APIKeys().openai_api_key.get_secret_value(),
            temperature=0.0,
        ),
        verbose=True,
    )

    # Create a task that requires code execution
    data_analysis_task = Task(
        description="Install the numpy library and calculate the average age the array [11,22,33].",
        agent=coding_agent,
        expected_output="Average age of participants",
    )
    # ToDo
    #  1. Find out USDC address on Gnosis using Tavily as tool
    #  2. Install web3 package
    #  3. Call the smart contract and find out balance of address

    # Create a crew and add the task
    analysis_crew = Crew(agents=[coding_agent], tasks=[data_analysis_task])

    # Execute the crew
    result = analysis_crew.kickoff()

    print(result)
