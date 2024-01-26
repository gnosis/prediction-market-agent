import typer
from prediction_market_agent.markets.omen import (
    omen_buy_outcome_tx,
    get_market,
    omen_sell_outcome_tx,
)
from prediction_market_agent.tools.types import (
    HexAddress,
    PrivateKey,
    xdai_type,
)

app = typer.Typer()


@app.command()
def buy(
    amount: str = typer.Option(),
    from_address: str = typer.Option(),
    from_private_key: str = typer.Option(),
    market_id: str = typer.Option(),
    outcome: str = typer.Option(),
    auto_deposit: bool = typer.Option(False),
) -> None:
    """
    Helper script to place a bet on Omen, usage:

    ```bash
    python scripts/bet_omen.py buy \
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
        amount=xdai_type(amount),
        from_address=HexAddress(from_address),
        from_private_key=PrivateKey(from_private_key),
        market=market,
        outcome=outcome,
        auto_deposit=auto_deposit,
    )


@app.command()
def sell(
    amount: str = typer.Option(),
    from_address: str = typer.Option(),
    from_private_key: str = typer.Option(),
    market_id: str = typer.Option(),
    outcome: str = typer.Option(),
    auto_withdraw: bool = typer.Option(False),
) -> None:
    """
    Helper script to sell outcome of an existing bet on Omen, usage:

    ```bash
    python scripts/bet_omen.py sell \
        --amount 0.01 \
        --from-address your-address \
        --from-private-key your-private-key \
        --market-id some-market-id \
        --outcome one-of-the-outcomes
    ```

    Market ID can be found easily in the URL: https://aiomen.eth.limo/#/0x86376012a5185f484ec33429cadfa00a8052d9d4
    """
    market = get_market(market_id)
    omen_sell_outcome_tx(
        amount=xdai_type(amount),
        from_address=HexAddress(from_address),
        from_private_key=PrivateKey(from_private_key),
        market=market,
        outcome=outcome,
        auto_withdraw=auto_withdraw,
    )


if __name__ == "__main__":
    app()
