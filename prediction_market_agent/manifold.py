import requests

from prediction_market_agent import utils
from prediction_market_agent.data_models.market_data_models import Market

"""
Python API for Manifold Markets

https://docs.manifold.markets/api#get-v0search-markets

Note: There is an existing wrapper here: https://github.com/vluzko/manifoldpy. Consider using that instead.
"""


def pick_binary_market() -> Market:
    url = "https://api.manifold.markets/v0/search-markets"
    params = {
        "term": "",
        "sort": "liquidity",
        "contractType": "BINARY",
        "limit": 1,
        "topicSlug": "forecaster-bot-war",
    }
    response = requests.get(url, params=params)

    response.raise_for_status()
    data = response.json()
    return Market(
        id=data[0]["id"],
        question=data[0]["question"],
        liquidity=float(data[0]["totalLiquidity"]),
    )


def place_bet(amount: int, market_id: str, outcome: bool, api_key):
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


if __name__ == "__main__":
    # A test run
    market = pick_binary_market()
    print(market.question)
    print("Placing bet on market:", market.question)
    place_bet(2, market.id, True, utils.get_manifold_api_key())
