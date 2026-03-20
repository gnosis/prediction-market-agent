# import csv
# from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
# from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy


# def main(limit: int = 2000, out_path: str = "open_omen_markets.csv") -> None:  # CHANGED: renamed output file to reflect OPEN markets
#     handler = OmenSubgraphHandler()

#     markets = handler.get_omen_markets_simple(
#         limit=limit,
#         filter_by=FilterBy.RESOLVED,  # CHANGED: now pulling RESOLVED markets instead of OPEN
#         sort_by=SortBy.NEWEST,
#     )

#     rows = []
#     p_count = 0  # CHANGED: count rows with p_yes (not Brier usable rows)

#     for m in markets:
#         md = m.model_dump()

#         outcomes = md.get("outcomes") or []
#         prices = md.get("outcomeTokenMarginalPrices") or []

#         # Find YES index
#         outcomes_upper = [str(o).upper() for o in outcomes]
#         yes_idx = outcomes_upper.index("YES") if "YES" in outcomes_upper else 0

#         # Get probability of YES
#         p_yes = None
#         if len(prices) > yes_idx and isinstance(prices[yes_idx], dict):
#             p_yes = prices[yes_idx].get("value")

#         if p_yes is not None:
#             p_count += 1  # CHANGED: track p_yes availability instead of y_yes

#         rows.append({
#             "market_id": md.get("id"),
#             "title": md.get("title"),
#             "p_yes": p_yes,
#             "collateralVolume": md.get("collateralVolume"),
#             "usdVolume": md.get("usdVolume"),
#             "creationTimestamp": md.get("creationTimestamp"),
#             "category": md.get("category"),
#         })  # CHANGED: removed y_yes completely (open markets don't have outcomes)

#     if rows:
#         with open(out_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(rows)

#     print(f"Wrote {len(rows)} rows → {out_path}. Rows with p_yes: {p_count}")  # CHANGED: updated print message


# if __name__ == "__main__":
#     main()



import csv
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy


def decode_answer_to_index(raw_answer: str | None) -> int | None:
    """
    Omen answers often come back as bytes32 hex like:
    0x000...0001, 0x000...0002, etc.
    Convert that to an integer index when possible.
    """
    if raw_answer is None:
        return None

    try:
        if isinstance(raw_answer, str) and raw_answer.startswith("0x"):
            val = int(raw_answer, 16)
            return val
    except Exception:
        pass

    return None


def main(limit: int = 2000, out_path: str = "resolved_omen_markets.csv") -> None:
    handler = OmenSubgraphHandler()

    markets = handler.get_omen_markets_simple(
        limit=limit,
        filter_by=FilterBy.RESOLVED,
        sort_by=SortBy.NEWEST,
    )

    rows = []

    for m in markets:
        md = m.model_dump()

        outcomes = md.get("outcomes") or []
        prices = md.get("outcomeTokenMarginalPrices") or []

        outcomes_upper = [str(o).upper() for o in outcomes]
        yes_idx = outcomes_upper.index("YES") if "YES" in outcomes_upper else None
        no_idx = outcomes_upper.index("NO") if "NO" in outcomes_upper else None

        p_yes = None
        if yes_idx is not None and len(prices) > yes_idx and isinstance(prices[yes_idx], dict):
            p_yes = prices[yes_idx].get("value")

        current_answer = md.get("currentAnswer")
        finalized_answer = md.get("finalizedAnswer")
        answer = md.get("answer")
        resolution = md.get("resolution")

        raw_outcome = finalized_answer or current_answer or answer or resolution
        answer_idx = decode_answer_to_index(raw_outcome)

        # In many Omen exports, answers are 1-based positions.
        # So 0x...01 means first outcome, 0x...02 means second outcome.
        winning_outcome = None
        outcome_yes = None

        if answer_idx is not None and answer_idx >= 1 and answer_idx <= len(outcomes):
            winning_outcome = outcomes[answer_idx - 1]
            winning_upper = str(winning_outcome).upper()

            if winning_upper == "YES":
                outcome_yes = 1
            elif winning_upper == "NO":
                outcome_yes = 0

        rows.append({
            "market_id": md.get("id"),
            "title": md.get("title"),
            "outcomes": "|".join(map(str, outcomes)),
            "p_yes": p_yes,
            "answer_idx": answer_idx,
            "winning_outcome": winning_outcome,
            "outcome_yes": outcome_yes,
            "raw_outcome": raw_outcome,
            "currentAnswer": current_answer,
            "finalizedAnswer": finalized_answer,
            "answer": answer,
            "resolution": resolution,
            "isResolved": md.get("isResolved"),
            "collateralVolume": md.get("collateralVolume"),
            "usdVolume": md.get("usdVolume"),
            "creationTimestamp": md.get("creationTimestamp"),
            "category": md.get("category"),
        })

    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    resolved_count = sum(r["outcome_yes"] is not None for r in rows)
    print(f"Wrote {len(rows)} rows → {out_path}")
    print(f"Rows with decoded YES/NO outcome: {resolved_count}")


if __name__ == "__main__":
    main()