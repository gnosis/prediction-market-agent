from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    PineconeMetadata,
)


def create_texts_from_omen_markets(markets: list[OmenMarket]) -> list[str]:
    return [m.question_title for m in markets]


def create_metadatas_from_omen_markets(markets: list[OmenMarket]):
    return [
        PineconeMetadata.model_validate(
            {"question_title": m.question_title, "market_address": m.id}
        ).dict()
        for m in markets
    ]
