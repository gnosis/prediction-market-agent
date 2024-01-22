import typer
from prediction_market_agent.omen import omen_buy_outcome_tx, get_market
from prediction_market_agent.tools.types import xDai, HexAddress, PrivateKey


def main(
    amount: str = typer.Option(),
    from_address: str = typer.Option(),
    from_private_key: str = typer.Option(),
    market_id: str = typer.Option(),
    outcome: str = typer.Option(),
) -> None:
    """
    Helper script to place a bet on Omen, usage:

    ```bash
    python scripts/bet_omen.py \
        --amount 0.01 \
        --from-address your-address \
        --from-private-key your-private-key \
        --market-id some-market-id \
        --outcome one-of-the-outcomes
    ```

    Market ID can be found easily in the URL: https://aiomen.eth.limo/#/0x86376012a5185f484ec33429cadfa00a8052d9d4
    """
    market = get_market(market_id)
    omen_buy_outcome_tx(
        amount=xDai(amount),
        from_address=HexAddress(from_address),
        from_private_key=PrivateKey(from_private_key),
        market=market,
        outcome=outcome,
        auto_deposit=True,
    )


if __name__ == "__main__":
    typer.run(main)
