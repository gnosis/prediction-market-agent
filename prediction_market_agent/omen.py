"""
Python API for Omen prediction market.

Their API is available as graph on https://thegraph.com/explorer/subgraphs/9V1aHHwkK4uPWgBH6ZLzwqFEkoHTHPS7XHKyjZWe8tEf?view=Overview&chain=mainnet,
but to not use our own credits, seems we can use their api deployment directly: https://api.thegraph.com/subgraphs/name/protofire/omen-xdai/graphql (link to the online playground)
"""
import os
import requests
from typing import Optional
from web3 import Web3
from web3.types import TxReceipt, TxParams
from pprint import pprint
from prediction_market_agent.data_models.market_data_models import Market
from prediction_market_agent.tools.web3_utils import (
    call_function_on_contract,
    call_function_on_contract_with_tx,
    WXDAI_ABI,
    xdai_to_wei,
    remove_fraction_wei,
    check_tx_receipt,
    ONE_NONCE,
)
from prediction_market_agent.tools.gnosis_rpc import GNOSIS_RPC_URL
from prediction_market_agent.tools.types import ABI, HexAddress, PrivateKey, xDai, Wei

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "abis/omen_fpmm.abi.json")
) as f:
    # File content taken from https://github.com/protofire/omen-exchange/blob/master/app/src/abi/marketMaker.json.
    OMEN_FPMM_ABI = ABI(f.read())

THEGRAPH_QUERY_URL = "https://api.thegraph.com/subgraphs/name/protofire/omen-xdai"

_QUERY_GET_SINGLE_FIXED_PRODUCT_MARKET_MAKER = """
query getFixedProductMarketMaker($id: String!) {
    fixedProductMarketMaker(
        id: $id
    ) {
        id
        title
        collateralVolume
        usdVolume
        collateralToken
        outcomes
    }
}
"""
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
        collateralVolume
        usdVolume
        collateralToken
        outcomes
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
    return Market.model_validate(market)


def get_market(market_id: str) -> Market:
    market = requests.post(
        THEGRAPH_QUERY_URL,
        json={
            "query": _QUERY_GET_SINGLE_FIXED_PRODUCT_MARKET_MAKER,
            "variables": {
                "id": market_id,
            },
        },
        headers={"Content-Type": "application/json"},
    ).json()["data"]["fixedProductMarketMaker"]
    return Market.model_validate(market)


def omen_approve_market_maker_to_spend_collateral_token(
    web3: Web3,
    market: Market,
    amount_wei: Wei,
    from_address: HexAddress,
    from_private_key: PrivateKey,
    tx_params: Optional[TxParams] = None,
) -> TxReceipt:
    return call_function_on_contract_with_tx(
        web3=web3,
        contract_address=market.collateral_token_contract_address_checksummed,
        contract_abi=WXDAI_ABI,
        from_address=from_address,
        from_private_key=from_private_key,
        function_name="approve",
        function_params=[
            market.market_maker_contract_address_checksummed,
            amount_wei,
        ],
        tx_params=tx_params,
    )


def omen_deposit_collateral_token(
    web3: Web3,
    market: Market,
    amount_wei: Wei,
    from_address: HexAddress,
    from_private_key: PrivateKey,
    tx_params: Optional[TxParams] = None,
) -> TxReceipt:
    return call_function_on_contract_with_tx(
        web3=web3,
        contract_address=market.collateral_token_contract_address_checksummed,
        contract_abi=WXDAI_ABI,
        from_address=from_address,
        from_private_key=from_private_key,
        function_name="deposit",
        tx_params={"value": amount_wei, **(tx_params or {})},
    )


def omen_calculate_buy_amount(
    web3: Web3,
    market: Market,
    investment_amount: Wei,
    outcome_index: int,
) -> Wei:
    """
    Returns amount of shares we will get for the given outcome_index for the given investment amount.
    """
    calculated_shares: Wei = call_function_on_contract(
        web3,
        market.market_maker_contract_address_checksummed,
        OMEN_FPMM_ABI,
        "calcBuyAmount",
        [investment_amount, outcome_index],
    )
    # Allow 1% slippage.
    min_outcome_tokens_to_buy = remove_fraction_wei(calculated_shares, 0.01)
    return min_outcome_tokens_to_buy


def omen_buy_shares(
    web3: Web3,
    market: Market,
    amount_wei: Wei,
    outcome_index: int,
    min_outcome_tokens_to_buy: Wei,
    from_address: HexAddress,
    from_private_key: PrivateKey,
    tx_params: Optional[TxParams] = None,
) -> TxReceipt:
    return call_function_on_contract_with_tx(
        web3=web3,
        contract_address=market.market_maker_contract_address_checksummed,
        contract_abi=OMEN_FPMM_ABI,
        from_address=from_address,
        from_private_key=from_private_key,
        function_name="buy",
        function_params=[
            amount_wei,
            outcome_index,
            min_outcome_tokens_to_buy,
        ],
        tx_params=tx_params,
    )


def omen_buy_outcome(
    amount: xDai,
    from_address: HexAddress,
    from_private_key: PrivateKey,
    market: Market,
    outcome: str,
    auto_deposit: bool,
) -> None:
    """
    Bets the given amount of xDai for the given outcome in the given market.
    """
    web3 = Web3(Web3.HTTPProvider(GNOSIS_RPC_URL))
    amount_wei = xdai_to_wei(amount)

    # Get the index of the outcome we want to buy.
    outcome_index: int = market.outcomes.index(outcome)

    # Get the current nonce for the given from_address.
    # If making multiple transactions quickly after each other,
    # it's better to increae it manually (otherwise we could get stale value from the network and error out).
    nonce = web3.eth.get_transaction_count(from_address)

    # Calculate the amount of shares we will get for the given investment amount.
    expected_shares = omen_calculate_buy_amount(web3, market, amount_wei, outcome_index)
    # Approve the market maker to withdraw our collateral token.
    approve_tx_receipt = omen_approve_market_maker_to_spend_collateral_token(
        web3=web3,
        market=market,
        amount_wei=amount_wei,
        from_address=from_address,
        from_private_key=from_private_key,
        tx_params={"nonce": nonce},
    )
    nonce += ONE_NONCE  # Increase after each tx.
    check_tx_receipt(approve_tx_receipt)
    # Deposit xDai to the collateral token,
    # this can be skipped, if we know we already have enough collateral tokens.
    if auto_deposit:
        deposit_receipt = omen_deposit_collateral_token(
            web3=web3,
            market=market,
            amount_wei=amount_wei,
            from_address=from_address,
            from_private_key=from_private_key,
            tx_params={"nonce": nonce},
        )
        nonce += ONE_NONCE  # Increase after each tx.
        check_tx_receipt(deposit_receipt)
    # Buy shares using the deposited xDai in the collateral token.
    buy_receipt = omen_buy_shares(
        web3,
        market,
        amount_wei,
        outcome_index,
        expected_shares,
        from_address,
        from_private_key,
        tx_params={"nonce": nonce},
    )
    nonce += ONE_NONCE  # Increase after each tx.
    check_tx_receipt(buy_receipt)


def omen_sell_outcome(market: Market) -> None:
    # TODO in next MR.
    ...


if __name__ == "__main__":
    pprint(get_omen_binary_markets(3))
