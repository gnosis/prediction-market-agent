import json
import re
from pathlib import Path
from typing import Any

import httpx
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.google_utils import search_google_serper
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_prophet.functions.web_scrape import web_scrape
from pydantic_ai.exceptions import UnexpectedModelBehavior
from sklearn.isotonic import IsotonicRegression


class Berlin1PolySentAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    LOG_PATH: Path | None = None

    def load(self) -> None:
        self.calibration_model = (
            train_calibration_model(self.LOG_PATH) if self.LOG_PATH else None
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        urls = search_google_serper(market.question)

        if not urls:
            logger.warning(f"No results found for {market.question}")
            return None

        contents = scrape_and_split_urls(urls)

        if not contents:
            logger.warning(f"No contents found for {market.question}")
            return None

        sentiment = extract_sentiment(contents)
        logger.info(f"Detected sentiment: {sentiment}")

        history_data = get_polymarket_history(market.question)
        history_summary = (
            summarize_history(history_data)
            if history_data
            else "No relevant historical data found."
        )

        probability, confidence = llm(
            market.question, contents, history_summary, sentiment
        )

        if self.calibration_model:
            try:
                probability = self.calibration_model.predict([probability])[0]
            except Exception as e:
                print(f"Calibration model failed: {e}")

        if sentiment == "Bullish":
            confidence = min(confidence + 0.05, 1.0)
        elif sentiment == "Bearish":
            confidence = max(confidence - 0.05, 0.0)

        probability, confidence = calibrate_probability(probability, confidence)

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
        )


def train_calibration_model(path: Path) -> IsotonicRegression:
    path.mkdir(exist_ok=True, parents=True)

    X, y = [], []

    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("resolved"):  # only use resolved markets
                X.append(entry["probability"])
                y.append(entry["resolved_value"])

    if len(X) < 5:
        raise ValueError("Not enough of data to calibrate.")

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(X, y)

    return model


def scrape_and_split_urls(urls: list[str]) -> list[str]:
    split_contents = []

    for url in urls:
        content = web_scrape(url)
        if not content:
            continue
        split_contents.extend(split_scraped_content(content[:10000]))

    return split_contents


def split_scraped_content(content: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(content)


def get_polymarket_history(market_question: str) -> list[dict[str, Any]]:
    # Construct the URL with your API key
    url = f"https://gateway.thegraph.com/api/{APIKeys().graph_api_key.get_secret_value()}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    query = f"""
    {{
      binaryMarkets(first: 5, where: {{question_contains: \"{market_question}\"}}) {{
        id
        question
        outcomes {{
          name
          price
        }}
        volume
        startTime
        endTime
      }}
    }}
    """

    response = httpx.post(url, json={"query": query})
    response.raise_for_status()

    json_data = response.json()
    data = json_data.get("data")

    if data is None:
        logger.warning(
            f"No 'data' found in response: {json_data}. Continuing without market history."
        )
        return []

    markets: list[dict[str, Any]] | None = data.get("binaryMarkets")

    if markets is None:
        logger.warning(
            f"No 'binaryMarkets' found in data: {json_data}. Continuing without market history."
        )
        return []

    return markets


@observe()
def summarize_history(history_data: list[dict[str, Any]]) -> str:
    summaries = []
    for market in history_data:
        outcomes = ", ".join(
            [f"{o['name']} ({float(o['price']):.2f})" for o in market["outcomes"]]
        )
        summaries.append(
            f"Market: {market['question']}, Outcomes: {outcomes}, Volume: {market['volume']}"
        )
    return "\n".join(summaries)


@observe()
def extract_sentiment(contents: list[str]) -> str:
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=APIKeys().openai_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a financial analyst skilled in understanding sentiment in news and analysis.",
            ),
            (
                "user",
                """Based on the following content, classify the overall market sentiment as one of:
- Bullish (positive/upbeat about YES outcome)
- Bearish (negative/downbeat about YES outcome)
- Neutral (mixed or uncertain)

Return only one word: Bullish, Bearish, or Neutral.

Content:
{content}
""",
            ),
        ]
    )

    merged_content = "\n".join(contents[:10])
    messages = prompt.format_messages(content=merged_content)
    response = llm.invoke(messages, max_tokens=16)

    return str(response.content).strip()


@observe()
def llm(
    question: str,
    contents: list[str],
    history_summary: str = "",
    sentiment: str = "Neutral",
) -> tuple[float, float]:
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=APIKeys().openai_api_key,
        temperature=0.5,
    )

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a professional prediction market trading agent. You combine market history, news analysis, sentiment, and judgment to make informed probabilistic forecasts.",
            ),
            (
                "user",
                """Today is {today}.

You are evaluating the following prediction market question:
**{question}**

Use the following steps:

1. Summarize key facts from the content.
2. Summarize relevant historical data.
3. Consider the sentiment.
4. Provide reasoning.
5. Estimate:
   - PROBABILITY: float between 0 and 1 (likelihood of YES)
   - CONFIDENCE: float between 0 and 1 (how strong the evidence is)

Respond in this format:

SUMMARY:
- ...
HISTORICAL SIGNALS:
- ...
SENTIMENT:
- ...
REASONING:
...

PROBABILITY: <float>
CONFIDENCE: <float>

Content:
{contents}

Historical Summary:
{history}

Market Sentiment:
{sentiment}
""",
            ),
        ]
    )

    messages = prompt.format_messages(
        today=utcnow(),
        question=question,
        contents=contents,
        history=history_summary,
        sentiment=sentiment,
    )
    response = llm.invoke(messages, max_tokens=1024)
    content = str(response.content)

    match = re.search(
        r"PROBABILITY:\s*([0-9.]+)\s*CONFIDENCE:\s*([0-9.]+)", content, re.IGNORECASE
    )
    if not match:
        raise UnexpectedModelBehavior(f"Could not parse LLM output: {content}")

    prob, conf = map(float, match.groups())
    return prob, conf


@observe()
def calibrate_probability(probability: float, confidence: float) -> tuple[float, float]:
    return probability * 0.95, min(confidence * 1.05, 1.0)
