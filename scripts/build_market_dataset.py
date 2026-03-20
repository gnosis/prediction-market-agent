import pandas as pd

from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler


def build_resolved_omen_snapshot_df(limit: int = 2000) -> pd.DataFrame:
    """
    Build one-row-per-market dataset for historical market analysis.

    Output columns:
      - market_id
      - question
      - category
      - created_time
      - close_time
      - finalized_time
      - resolution
      - prob_at_close
      - time_to_resolution_hours
      - volume
      - url
    """

    handler = OmenSubgraphHandler()

    # Pull resolved binary markets.
    # In this repo, get_omen_markets_simple(filter_by=FilterBy.RESOLVED)
    # sets resolved=True, and the handler's binary defaults keep outcomeSlotCount=2.
    resolved_markets = handler.get_omen_markets_simple(
        limit=limit,
        filter_by=FilterBy.RESOLVED,
        sort_by=SortBy.NEWEST,
    )

    rows = []

    for raw_market in resolved_markets:
        try:
            # Convert to agent market so we can use get_last_trade_p_yes()
            market = OmenAgentMarket.from_data_model(raw_market)

            # Last market-implied YES probability before close
            prob_at_close = market.get_last_trade_p_yes()

            # Skip markets with no usable trade history
            if prob_at_close is None:
                continue

            # These resolved markets should already exclude invalid answers,
            # but we still guard just in case.
            resolution = raw_market.question.boolean_outcome

            time_to_resolution_hours = None
            if raw_market.finalized_datetime is not None:
                delta = raw_market.finalized_datetime - market.close_time
                time_to_resolution_hours = delta.total_seconds() / 3600

            rows.append(
                {
                    "market_id": market.id,
                    "question": market.question,
                    "category": raw_market.category,
                    "created_time": raw_market.creation_datetime,
                    "close_time": market.close_time,
                    "finalized_time": raw_market.finalized_datetime,
                    "resolution": int(resolution),   # YES=1, NO=0
                    "prob_at_close": float(prob_at_close),
                    "time_to_resolution_hours": time_to_resolution_hours,
                    "volume": float(market.volume),
                    "url": market.url,
                }
            )

        except Exception as e:
            print(f"Skipping market {getattr(raw_market, 'id', 'unknown')}: {e}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("close_time").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = build_resolved_omen_snapshot_df(limit=500)
    print(df.head())
    print(df.shape)
    df.to_csv("omen_resolved_snapshots.csv", index=False)
