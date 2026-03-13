import csv
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy


def main(limit: int = 2000, out_path: str = "open_omen_markets.csv") -> None:  # CHANGED: renamed output file to reflect OPEN markets
    handler = OmenSubgraphHandler()

    markets = handler.get_omen_markets_simple(
        limit=limit,
        filter_by=FilterBy.OPEN,  # CHANGED: now pulling OPEN markets instead of RESOLVED
        sort_by=SortBy.NEWEST,
    )

    rows = []
    p_count = 0  # CHANGED: count rows with p_yes (not Brier usable rows)

    for m in markets:
        md = m.model_dump()

        outcomes = md.get("outcomes") or []
        prices = md.get("outcomeTokenMarginalPrices") or []

        # Find YES index
        outcomes_upper = [str(o).upper() for o in outcomes]
        yes_idx = outcomes_upper.index("YES") if "YES" in outcomes_upper else 0

        # Get probability of YES
        p_yes = None
        if len(prices) > yes_idx and isinstance(prices[yes_idx], dict):
            p_yes = prices[yes_idx].get("value")

        if p_yes is not None:
            p_count += 1  # CHANGED: track p_yes availability instead of y_yes

        rows.append({
            "market_id": md.get("id"),
            "title": md.get("title"),
            "p_yes": p_yes,
            "collateralVolume": md.get("collateralVolume"),
            "usdVolume": md.get("usdVolume"),
            "creationTimestamp": md.get("creationTimestamp"),
            "category": md.get("category"),
        })  # CHANGED: removed y_yes completely (open markets don't have outcomes)

    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_path}. Rows with p_yes: {p_count}")  # CHANGED: updated print message


if __name__ == "__main__":
    main()
