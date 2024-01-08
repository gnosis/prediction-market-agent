import requests

from utils import get_manifold_api_key

"""
Python API for Manifold Markets

https://docs.manifold.markets/api#get-v0search-markets

Note: There is an existing wrapper here: https://github.com/vluzko/manifoldpy. Consider using that instead.
"""


class Market:
    def __init__(self, market_json: dict):
        self.question = market_json["question"]
        self.id = market_json["id"]
        self.liquidity = market_json["totalLiquidity"]

    def __repr__(self):
        return f"Market: {self.question}"


def pick_binary_market() -> Market:
    url = "https://api.manifold.markets/v0/search-markets"
    params = {
        "term": "",
        "sort": "liquidity",
        "filter": "closing-this-month",
        "contractType": "BINARY",
        "limit": 1,
    }
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        return Market(data[0])
    else:
        raise Exception(
            f"Picking market failed: {response.status_code} {response.reason} {response.text}"
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
    print("Placing bet on market:", market.question)
    place_bet(2, market.id, True, get_manifold_api_key())
