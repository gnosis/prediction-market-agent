import pandas as pd

from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler


def build_resolved_omen_snapshot_df(limit: int = 2000) -> pd.DataFrame:
    """
    Build a one-row-per-market dataset for historical market analysis.

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
      - duration_hours
      - volume
      - url
    """

    print("Initializing OmenSubgraphHandler...")
    handler = OmenSubgraphHandler()

    print(f"Fetching up to {limit} resolved binary markets...")
    resolved_markets = handler.get_omen_markets_simple(
        limit=limit,
        filter_by=FilterBy.RESOLVED,
        sort_by=SortBy.NEWEST,
    )
    print(f"Fetched {len(resolved_markets)} markets.")

    rows = []

    for i, raw_market in enumerate(resolved_markets, start=1):
        market_id = getattr(raw_market, "id", "unknown")

        if i % 50 == 0 or i == 1:
            print(f"Processing market {i}/{len(resolved_markets)}: {market_id}")

        try:
            market = OmenAgentMarket.from_data_model(raw_market)

            # Final pre-close YES probability
            prob_at_close = market.get_last_trade_p_yes()

            # Skip markets with no usable trade history
            if prob_at_close is None:
                continue

            created_time = raw_market.creation_datetime
            close_time = market.close_time
            finalized_time = raw_market.finalized_datetime

            # Binary resolved outcome: YES=1, NO=0
            resolution = raw_market.question.boolean_outcome
            if resolution is None:
                continue

            # How long after close the market was finalized
            time_to_resolution_hours = None
            if finalized_time is not None and close_time is not None:
                delta = finalized_time - close_time
                time_to_resolution_hours = delta.total_seconds() / 3600

            # Market lifetime from creation to close
            duration_hours = None
            if created_time is not None and close_time is not None:
                delta = close_time - created_time
                duration_hours = delta.total_seconds() / 3600

            rows.append(
                {
                    "market_id": market.id,
                    "question": market.question,
                    "category": raw_market.category,
                    "created_time": created_time,
                    "close_time": close_time,
                    "finalized_time": finalized_time,
                    "resolution": int(resolution),
                    "prob_at_close": float(prob_at_close),
                    "time_to_resolution_hours": time_to_resolution_hours,
                    "duration_hours": duration_hours,
                    "volume": float(market.volume),
                    "url": market.url,
                }
            )

        except Exception as e:
            print(f"Skipping market {market_id}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows collected.")
        return df

    # Convert datetime columns cleanly
    datetime_cols = ["created_time", "close_time", "finalized_time"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Basic cleanup
    df = df.dropna(subset=["market_id", "prob_at_close", "resolution"])
    df = df[(df["prob_at_close"] >= 0) & (df["prob_at_close"] <= 1)]

    df = df.sort_values("close_time").reset_index(drop=True)

    print("\nFinished dataset build.")
    print(f"Final shape: {df.shape}")
    print("\nResolution counts:")
    print(df["resolution"].value_counts(dropna=False))
    print("\nCategory counts:")
    print(df["category"].value_counts(dropna=False).head(10))

    return df


if __name__ == "__main__":
    df = build_resolved_omen_snapshot_df(limit=2000)
    print("\nPreview:")
    print(df.head())
    df.to_csv("omen_resolved_snapshots.csv", index=False)
    print("\nSaved to omen_resolved_snapshots.csv")
