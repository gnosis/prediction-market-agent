from datetime import datetime, timedelta

from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.gtypes import ChecksumAddress, wei_type, xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.categorize import infer_category
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from prediction_market_agent_tooling.markets.omen.data_models import (
    OMEN_FALSE_OUTCOME,
    OMEN_TRUE_OUTCOME,
)
from prediction_market_agent_tooling.markets.omen.omen import (
    OMEN_DEFAULT_MARKET_FEE,
    OmenAgentMarket,
    omen_create_market_tx,
    omen_remove_fund_market_tx,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.is_predictable import is_predictable_binary
from prediction_market_agent_tooling.tools.utils import utcnow

# According to Omen's recommendation, closing time of the market should be at least 6 days after the outcome is known.
# That is because at the closing time, the question will open on Realitio, and we don't want it to be resolved as unknown/invalid.
# All replicated markets that close at N, needs to have closing time on Realition N + `EXTEND_CLOSING_TIME_DELTA`.
EXTEND_CLOSING_TIME_DELTA = timedelta(days=6)


def omen_replicate_from_tx(
    private_credentials: PrivateCredentials,
    market_type: MarketType,
    n_to_replicate: int,
    initial_funds: xDai,
    close_time_before: datetime | None = None,
    auto_deposit: bool = False,
) -> list[ChecksumAddress]:
    from_address = private_credentials.public_key
    already_created_markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=from_address,
    )

    markets = get_binary_markets(
        # Polymarket is slow to get, so take only 10 candidates for him.
        10 if market_type == MarketType.POLYMARKET else 100,
        market_type,
        filter_by=FilterBy.OPEN,
        sort_by=SortBy.NONE,
        excluded_questions=set(m.question_title for m in already_created_markets),
    )
    markets_sorted = sorted(
        markets,
        key=lambda m: m.volume or 0,
        reverse=True,
    )
    markets_to_replicate = [
        m
        for m in markets_sorted
        if close_time_before is None
        or (m.close_time is not None and m.close_time <= close_time_before)
    ]
    if not markets_to_replicate:
        logger.info(f"No markets found for {market_type}")
        return []

    logger.info(f"Found {len(markets_to_replicate)} markets to replicate.")

    # Get a set of possible categories from existing markets (but created by anyone, not just your agent)
    existing_categories = set(
        m.category
        for m in OmenSubgraphHandler().get_omen_binary_markets_simple(
            limit=1000,
            sort_by=SortBy.NEWEST,
            filter_by=FilterBy.NONE,
        )
    )

    created_addresses: list[ChecksumAddress] = []

    for market in markets_to_replicate:
        if market.close_time is None:
            logger.info(
                f"Skipping `{market.question}` because it's missing the closing time."
            )
            continue

        safe_closing_time = market.close_time + EXTEND_CLOSING_TIME_DELTA
        # Force at least 48 hours of time where the resolution is unknown.
        soonest_allowed_resolution_known_time = utcnow() + timedelta(hours=48)
        if market.close_time <= soonest_allowed_resolution_known_time:
            logger.info(
                f"Skipping `{market.question}` because it closes sooner than {soonest_allowed_resolution_known_time}."
            )
            continue

        # Do as the last step, becuase it calls OpenAI (costly & slow).
        if not is_predictable_binary(market.question):
            logger.info(
                f"Skipping `{market.question}` because it seems to not be predictable."
            )
            continue

        category = infer_category(market.question, existing_categories)
        # Realitio will allow new categories or misformated categories, so double check that the LLM got it right.
        if category not in existing_categories:
            logger.info(
                f"Error: LLM went rouge. Skipping `{market.question}` because the category `{category}` is not in the existing categories {existing_categories}."
            )
            continue

        market_address = omen_create_market_tx(
            private_credentials=private_credentials,
            initial_funds=initial_funds,
            fee=OMEN_DEFAULT_MARKET_FEE,
            question=market.question,
            closing_time=safe_closing_time,
            category=category,
            language="en",
            outcomes=[OMEN_TRUE_OUTCOME, OMEN_FALSE_OUTCOME],
            auto_deposit=auto_deposit,
        )
        created_addresses.append(market_address)
        logger.info(
            f"Created `https://aiomen.eth.limo/#/{market_address}` for `{market.question}` in category {category} out of {market.url}."
        )

        if len(created_addresses) >= n_to_replicate:
            logger.info(
                f"Replicated {len(created_addresses)} from {market_type}, breaking."
            )
            break

    return created_addresses


def omen_unfund_replicated_known_markets_tx(
    private_credentials: PrivateCredentials,
    saturation_above_threshold: float | None = None,
) -> None:
    from_address = private_credentials.public_key

    now = utcnow()
    # We want to unfund markets ~1 day before the resolution should be known.
    # That is, if the original market would be closing now, but we added `EXTEND_CLOSING_TIME_DELTA` to it,
    # we want to unfund any market that closes sooner than NOW + `EXTEND_CLOSING_TIME_DELTA` - 1 day.
    opened_before = now + EXTEND_CLOSING_TIME_DELTA - timedelta(days=1)

    # Fetch markets that we created, are soon to be known,
    # and still have liquidity in them (we didn't withdraw it yet).
    markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=from_address,
        opened_before=opened_before,
        liquidity_bigger_than=wei_type(0),
    )

    for idx, market in enumerate(markets):
        # Optionally, if `saturation_above_threshold` is provided, skip markets that are not saturated to leave some free money motivation for agents.
        if (
            saturation_above_threshold is not None
            and not market.is_resolved
            and not (
                market.current_p_yes > saturation_above_threshold
                or market.current_p_no > saturation_above_threshold
            )
        ):
            logger.info(
                f"[{idx+1}/{len(markets)}] Skipping unfunding of `{market.liquidityParameter=} {market.question=}  {market.url=}`, because it's not saturated yet, `{market.current_p_yes=}`."
            )
            continue
        logger.info(
            f"[{idx+1}/{len(markets)}] Unfunding market `{market.liquidityParameter=} {market.question=} {market.url=}`."
        )
        omen_remove_fund_market_tx(
            private_credentials=private_credentials,
            market=OmenAgentMarket.from_data_model(market),
            shares=None,
        )
