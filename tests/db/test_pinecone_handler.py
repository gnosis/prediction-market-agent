from typing import Generator
from unittest.mock import Mock, patch

import pytest
from eth_typing import HexAddress, HexStr
from langchain_chroma import Chroma

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    PineconeMetadata,
)
from prediction_market_agent.db.pinecone_handler import PineconeHandler

TRUMP_MARKETS = [
    "Will Donald Trump announce his vice presidential pick by 5 July 2024?",
    "Will Donald Trump and Ronald DeSantis join forces to run on the Republican joint-ticket for the 2024 United States presidential election?",
]
BIDEN_MARKETS = [
    "Will Joe Biden drop out of the presidential race on 8 July 2024?",
    "Will there be a call for President Biden's removal from office on 8 July 2024?",
    "Will President Joe Biden announce his intention not to run for a second term by 8 July 2024?",
]
UNRELATED_MARKETS = [
    "Will extreme heat persist in most parts of the US on 8 July 2024?",
    "Will Cristiano Ronaldo score in the Euro 2024 quarter-finals on 7 July 2024?",
]


@pytest.fixture()
def test_pinecone_handler() -> Generator[PineconeHandler, None, None]:
    with patch(
        "prediction_market_agent.db.pinecone_handler.PineconeHandler.build_vectorstore",
        Mock(return_value=None),
    ):
        p = PineconeHandler()

    p.vectorstore = Chroma(embedding_function=p.embeddings)
    texts = TRUMP_MARKETS + BIDEN_MARKETS + UNRELATED_MARKETS
    metadatas = [
        PineconeMetadata(
            question_title=text,
            market_address=HexAddress(HexStr("")),
            close_time_timestamp=0,
        ).model_dump()
        for text in texts
    ]
    p.insert_texts(
        ids=[p.encode_text(text) for text in texts], texts=texts, metadatas=metadatas
    )
    yield p


def test_search_similarity(test_pinecone_handler: PineconeHandler) -> None:
    limit = len(TRUMP_MARKETS) + len(BIDEN_MARKETS)
    # We aim to find all presidential-related questions - we add 1 to test the threshold effectiveness
    questions = test_pinecone_handler.find_nearest_questions_with_threshold(
        limit=limit + 1, text="Will Trump win the election in 2024?"
    )
    assert len(questions) == limit
