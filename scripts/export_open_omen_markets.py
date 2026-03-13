import csv
from datetime import datetime
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler


def get_value(x):
    """Convert typed dicts like {'value': 0.0, 'type': 'USD'} to plain numbers."""
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x


def main(limit: int = 500) -> None:
    h = OmenSubgraphHandler()

    # CHANGED: use full endpoint so liquidity fields are present
    markets = h.get_omen_markets(
        limit=limit,
        include_categorical_markets=False,
        include_scalar_markets=False,
    )

    today = datetime.utcnow().strftime("%Y-%m-%d")
    out_path = f"data/open_omen_markets_{today}.csv"

    rows = []
    p_count = 0

    for m in markets:
        md = m.model_dump()

        outcomes = md.get("outcomes") or []
        prices = md.get("outcomeTokenMarginalPrices") or []

        outcomes_upper = [str(o).upper() for o in outcomes]
        yes_idx = outcomes_upper.index("YES") if "YES" in outcomes_upper else 0

        p_yes = None
        if len(prices) > yes_idx:
            # prices might be dicts or raw floats depending on endpoint
            p_yes = get_value(prices[yes_idx])

        if p_yes is not None:
            p_count += 1

        rows.append({
            "market_id": md.get("id"),
            "title": md.get("title"),
            "p_yes": p_yes,
            "collateralVolume": get_value(md.get("collateralVolume")),   # CHANGED: parse typed dict
            "usdVolume": get_value(md.get("usdVolume")),                 # CHANGED: parse typed dict
            "liquidityParameter": get_value(md.get("liquidityParameter")), # CHANGED: added liquidity proxy
            "creationTimestamp": md.get("creationTimestamp"),
            "category": md.get("category"),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_path}")
    print(f"Rows with p_yes: {p_count}")


if __name__ == "__main__":
    main()
