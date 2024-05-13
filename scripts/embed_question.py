import sys

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)

from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.utils import APIKeys


def main():
    k = APIKeys()
    print("start")
    sh = OmenSubgraphHandler()
    open_markets = sh.get_omen_binary_markets_simple(
        limit=sys.maxsize, filter_by=FilterBy.OPEN, sort_by=SortBy.NEWEST
    )

    ph = PineconeHandler()
    texts = [m.question_title for m in open_markets]
    ph.insert_texts_if_not_exists(texts=texts)
    print("end")


if __name__ == "__main__":
    main()
