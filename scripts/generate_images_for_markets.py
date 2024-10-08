import typer
from prediction_market_agent_tooling.gtypes import HexAddress, HexStr, Wei
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenThumbnailMapping,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)

from prediction_market_agent.agents.replicate_to_omen_agent.image_gen import (
    generate_and_set_image_for_market,
)
from prediction_market_agent.utils import APIKeys


def main(creator: str) -> None:
    """
    Generates and sets images for all markets created by the given creator, if they still have some liquidity.
    """
    keys = APIKeys()
    image_contract = OmenThumbnailMapping()
    markets = [
        OmenAgentMarket.from_data_model(m)
        for m in OmenSubgraphHandler().get_omen_binary_markets(
            limit=None,
            creator=HexAddress(HexStr(creator)),
            liquidity_bigger_than=Wei(0),
        )
    ]

    for idx, market in enumerate(markets):
        if existing_image_url := image_contract.get_url(
            market.market_maker_contract_address_checksummed
        ):
            print(
                f"[{idx + 1} / {len(markets)}] Skipping {market.url} because it already has an image: {existing_image_url}."
            )
            continue

        if (
            generate_and_set_image_for_market(
                market.market_maker_contract_address_checksummed, market.question, keys
            )
            is None
        ):
            print(
                f"[{idx + 1} / {len(markets)}] Skipping {market.url} because of failed generation."
            )
            continue

        image_url = image_contract.get_url(
            market.market_maker_contract_address_checksummed
        )
        print(
            f"[{idx + 1} / {len(markets)}] Saved image for {market.url} at {image_url}."
        )


if __name__ == "__main__":
    typer.run(main)
