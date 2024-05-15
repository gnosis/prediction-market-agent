import sys

import typer
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    PineconeMetadata,
)
from prediction_market_agent.db.pinecone_handler import PineconeHandler


def main() -> None:
    """Script for inserting all open markets into Pinecone (if not yet there)."""
    sh = OmenSubgraphHandler()
    open_markets = sh.get_omen_binary_markets_simple(
        limit=sys.maxsize, filter_by=FilterBy.OPEN, sort_by=SortBy.NEWEST
    )

    ph = PineconeHandler()
    texts = [m.question_title for m in open_markets]
    metadatas = [
        PineconeMetadata.model_validate(
            {"question_title": m.question_title, "market_address": m.id}
        ).dict()
        for m in open_markets
    ]
    ph.insert_texts_if_not_exists(texts=texts, metadatas=metadatas)
    print("end")


if __name__ == "__main__":
    typer.run(main)
