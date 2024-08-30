import altair as alt
import pandas as pd
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.tools.cache import persistent_inmemory_cache
from prediction_market_agent_tooling.tools.utils import LLM_SUPER_LOW_TEMPERATURE


@persistent_inmemory_cache
def llm_random_numbers(
    n: int,
    engine: str,
    temperature: float,
    seed: int | None,
    trial: int,  # Used to invalidate cache between runs.
) -> list[int]:
    llm = ChatOpenAI(
        model=engine,
        temperature=temperature,
        seed=seed,
        api_key=APIKeys().openai_api_key_secretstr_v1,
    )
    prompt_template = "Generate {n} random numbers between 1 and 100. Return only them, no additional text, write them comma-separated."
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(n=n)
    completion = [
        int(x) for x in str(llm.invoke(messages, max_tokens=512).content).split(",")
    ]
    return completion


st.set_page_config(page_title="LLM Randomness", layout="wide")

trials = st.number_input("How many trials do you want to run?", value=5)
n = st.number_input("How many random numbers do you want to generate?", value=20)
engines = [
    e.strip()
    for e in st.text_input(
        "Engines (comma-separated)",
        "gpt-3.5-turbo, gpt-4o-2024-08-06, gpt-4-1106-preview, gpt-4-turbo-2024-04-09",
    ).split(",")
]
seed = (
    st.number_input("Seed", value=0) if st.checkbox("Use seed", value=False) else None
)
temperature = float(st.selectbox("Temperature", [0.0, 1.0, LLM_SUPER_LOW_TEMPERATURE]))

st.header(f"Temperature {temperature} Seed {seed}")

for col, engine in zip(st.columns(len(engines)), engines):
    all_data: list[pd.DataFrame] = []

    for trial in range(trials):
        numbers = llm_random_numbers(
            n=n, engine=engine, temperature=temperature, seed=seed, trial=trial
        )
        trial_data = pd.DataFrame(
            {
                "Position": range(1, len(numbers) + 1),
                "Random Number": numbers,
                "Trial": [f"Trial {trial + 1}"] * len(numbers),
            }
        )
        all_data.append(trial_data)

    combined_data = pd.concat(all_data, ignore_index=True)

    chart = (
        alt.Chart(combined_data)
        .mark_bar()
        .encode(
            x=alt.X("Position:O", axis=alt.Axis(title="Position")),
            y=alt.Y("Random Number:Q", axis=alt.Axis(title="Random Number")),
            color="Trial:N",
            xOffset="Trial:N",
        )
        .properties(width=600, height=400)
    )

    with col:
        st.header(f"Engine {engine}")
        st.altair_chart(chart)
