"""
PYTHONPATH=. streamlit run scripts/image_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import streamlit as st
from openai import OpenAI
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from prediction_market_agent_tooling.tools.streamlit_user_login import streamlit_login

st.title("Prediction market thumbnail generator")

with st.sidebar:
    streamlit_login()

markets = get_binary_markets(42, MarketType.OMEN)

custom_question_input = st.checkbox("Provide a custom question", value=False)
question = (
    st.text_input("Question")
    if custom_question_input
    else st.selectbox("Select a question", [m.question for m in markets])
)

prompt_template = st.text_area(
    "Prompt",
    value="""Create a thumbnail image for the following prediction market '{market}'.
Don't write any text on the image.""",
)

market = (
    [m for m in markets if m.question == question][0]
    if not custom_question_input
    # If custom question is provided, just take some random market and update its question.
    else markets[0].model_copy(update={"question": question, "current_p_yes": 0.5})
)

with st.spinner("Generating image..."):
    response = OpenAI().images.generate(
        model="dall-e-3",
        prompt=prompt_template.format(market=market.question),
        size="1024x1024",
        quality="standard",
        n=1,
    )
image_url = response.data[0].url

if image_url is not None:
    st.image(image_url, use_column_width=True)
else:
    st.error("Failed to generate image.")
