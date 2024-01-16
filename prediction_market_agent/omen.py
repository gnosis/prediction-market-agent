"""
Python API for Omen prediction market.

Their API is available as graph on https://thegraph.com/explorer/subgraphs/9V1aHHwkK4uPWgBH6ZLzwqFEkoHTHPS7XHKyjZWe8tEf?view=Overview&chain=mainnet,
but to not use our own credits, seems we can use their api deployment directly: https://api.thegraph.com/subgraphs/name/protofire/omen-xdai/graphql (link to the online playground)
"""
import requests

from prediction_market_agent.models import Market

THEGRAPH_QUERY_URL = "https://api.thegraph.com/subgraphs/name/protofire/omen-xdai"

# Honestly modified from https://github.com/valory-xyz/trader/blob/main/packages/valory/skills/market_manager_abci/graph_tooling/queries/omen.py
_QUERY_GET_FIXED_PRODUCT_MARKETS_MAKERS = """
query getFixedProductMarketMakers($first: Int!, $outcomes: [String!]) {
    fixedProductMarketMakers(
        where: {
            isPendingArbitration: false,
            outcomes: $outcomes
        },
        orderBy: creationTimestamp,
        orderDirection: desc,
        first: $first
    ) {
        id
        title
        scaledLiquidityParameter
    }
}
"""


def get_omen_markets(first: int, outcomes: list[str]) -> dict:
    markets = requests.post(
        THEGRAPH_QUERY_URL,
        json={
            "query": _QUERY_GET_FIXED_PRODUCT_MARKETS_MAKERS,
            "variables": {
                "first": first,
                "outcomes": outcomes,
            },
        },
        headers={"Content-Type": "application/json"},
    ).json()["data"]["fixedProductMarketMakers"]
    return markets


def get_omen_binary_markets(first: int) -> dict:
    return get_omen_markets(first, ["Yes", "No"])


def pick_binary_market() -> Market:
    market = get_omen_binary_markets(first=1)[0]
    return Market(
        id=market["id"],
        question=market["title"],
        liquidity=float(market["scaledLiquidityParameter"]),
    )


def place_bet(amount: int, market_id: str, outcome: bool, api_key: str):
    outcome_str = "YES" if outcome else "NO"
    url = "https://api.manifold.markets/v0/bet"
    params = {
        "amount": amount,
        "contractId": market_id,
        "outcome": outcome_str,
    }

    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        assert data["isFilled"]
    else:
        raise Exception(
            f"Placing bet failed: {response.status_code} {response.reason} {response.text}"
        )
