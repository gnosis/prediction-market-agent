from datetime import timedelta

from prediction_market_agent_tooling.gtypes import (
    USD,
    ChecksumAddress,
    CollateralToken,
    HexAddress,
    HexStr,
    Wei,
    int_to_hexbytes,
)
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
    OMEN_DEFAULT_MARKET_FEE_PERC,
    OmenAgentMarket,
    omen_create_market_tx,
    omen_remove_fund_market_tx,
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.is_invalid import is_invalid
from prediction_market_agent_tooling.tools.is_predictable import (
    is_predictable_binary,
    is_predictable_without_description,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import DatetimeUTC, utcnow

from prediction_market_agent.agents.replicate_to_omen_agent.image_gen import (
    generate_and_set_image_for_market,
)
from prediction_market_agent.agents.replicate_to_omen_agent.rephrase import rephrase
from prediction_market_agent.db.models import ReplicatedMarket
from prediction_market_agent.db.replicated_markets_table_handler import (
    ReplicatedMarketsTableHandler,
)
from prediction_market_agent.utils import APIKeys

# According to Omen's recommendation, closing time of the market should be at least 6 days after the outcome is known.
# That is because at the closing time, the question will open on Realitio, and we don't want it to be resolved as unknown/invalid.
# All replicated markets that close at N, needs to have closing time on Realition N + `EXTEND_CLOSING_TIME_DELTA`.
EXTEND_CLOSING_TIME_DELTA = timedelta(days=6)


@observe()
def omen_replicate_from_tx(
    api_keys: APIKeys,
    market_type: MarketType,
    n_to_replicate: int,
    initial_funds: USD | CollateralToken,
    collateral_token_address: ChecksumAddress,
    replicated_market_table_handler: ReplicatedMarketsTableHandler,
    close_time_before: DatetimeUTC | None = None,
    close_time_after: DatetimeUTC | None = None,
    auto_deposit: bool = False,
    test: bool = False,
) -> list[ChecksumAddress]:
    existing_markets = OmenSubgraphHandler().get_omen_markets(limit=None)

    replicated_markets = (
        replicated_market_table_handler.get_replicated_markets_from_market(
            parent_market_type=market_type
        )
    )

    excluded_questions = set(
        [m.question_title for m in existing_markets]
        + [i.original_market_title for i in replicated_markets]
        + [i.copied_market_title for i in replicated_markets]
    )

    markets = get_binary_markets(
        500 if market_type == MarketType.POLYMARKET else 1000,
        market_type,
        filter_by=FilterBy.OPEN,
        sort_by=(
            SortBy.HIGHEST_LIQUIDITY
            if market_type == MarketType.POLYMARKET
            else SortBy.CLOSING_SOONEST
        ),
        excluded_questions=excluded_questions,
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
        or m.question in excluded_questions
    ]
    if not markets_to_replicate:
        logger.info(f"No markets found for {market_type}")
        return []

    logger.info(f"Found {len(markets_to_replicate)} markets to replicate.")

    # Get a set of possible categories from existing markets (but created by anyone, not just your agent)
    existing_categories = set(m.category for m in existing_markets)

    created_addresses: list[ChecksumAddress] = []
    created_questions: set[str] = set()

    for market in markets_to_replicate:
        original_market_question = market.question
        # We initially consider that market does not need to be rephrased.

        if len(created_addresses) >= n_to_replicate:
            logger.info(
                f"Replicated {len(created_addresses)} from {market_type}, breaking."
            )
            break

        if market.question in created_questions:
            logger.info(
                f"Skipping `{market.question}` because it was already replicated in this run."
            )
            continue

        if market.close_time is None:
            logger.info(
                f"Skipping `{market.question}` because it's missing the closing time."
            )
            continue

        safe_closing_time = market.close_time + EXTEND_CLOSING_TIME_DELTA
        # If `close_time_after` isn't provided, force at least 48 hours of time where the resolution is unknown.
        soonest_allowed_resolution_known_time = (
            close_time_after
            if close_time_after is not None
            else utcnow() + timedelta(hours=48)
        )
        if market.close_time <= soonest_allowed_resolution_known_time:
            logger.info(
                f"Skipping `{market.question}` because it closes sooner than {soonest_allowed_resolution_known_time}."
            )
            continue

        # Do as the last steps, because it calls OpenAI (costly & slow).
        if is_invalid(market.question):
            logger.info(
                f"Skipping `{market.question}` was marked as invalid. Trying to rephrase and make it valid."
            )
            # We try rephrasing the question to make it valid, and run the validity check again.
            new_question = rephrase(market.question)
            logger.info(f"Rephrased `{market.question}` to `{new_question}`.")
            if is_invalid(new_question):
                logger.info(
                    f"Skipping `{new_question}` because it could not be rephrased into a valid question."
                )
                continue
            else:
                market.question = new_question

        if not is_predictable_binary(market.question):
            logger.info(
                f"Skipping `{market.question}` because it seems to not be predictable."
            )
            continue

        if market.description and not is_predictable_without_description(
            market.question, market.description
        ):
            # We try rephrasing the question to combine elements of the description into the question.
            new_question = rephrase(market.question + market.description)
            logger.info(
                f"Rephrased `{market.question}` to `{new_question}` with the description `{market.description}`."
            )
            if not is_predictable_without_description(new_question, market.description):
                logger.info(
                    f"Skipping `{market.question}` because it could not be rephrased into a valid question without the description `{market.description}`. The rephrased question was `{new_question}`."
                )
                continue
            else:
                market.question = new_question

        category = infer_category(market.question, existing_categories)
        # Realitio will allow new categories or misformated categories, so double check that the LLM got it right.
        if category not in existing_categories:
            logger.info(
                f"Error: LLM went rouge. Skipping `{market.question}` because the category `{category}` is not in the existing categories {existing_categories}."
            )
            continue

        if test:
            logger.info(
                f"Test mode: Would create `{market.question}` in category {category} out of {market.url}."
            )
            created_addresses.append(
                ChecksumAddress(HexAddress(HexStr(int_to_hexbytes(0).hex())))
            )
            created_questions.add(market.question)
            continue

        logger.info(
            f"Replicating {market.question} from {market.url} in category {category}."
        )

        created_market = omen_create_market_tx(
            api_keys=api_keys,
            initial_funds=initial_funds,
            fee_perc=OMEN_DEFAULT_MARKET_FEE_PERC,
            question=market.question,
            closing_time=safe_closing_time,
            category=category,
            language="en",
            outcomes=[OMEN_TRUE_OUTCOME, OMEN_FALSE_OUTCOME],
            auto_deposit=auto_deposit,
            collateral_token_address=collateral_token_address,
        )
        market_address = (
            created_market.market_event.fixed_product_market_maker_checksummed
        )
        created_addresses.append(market_address)
        created_questions.add(market.question)

        replicated_market = ReplicatedMarket(
            original_market_type=market_type.value,
            original_market_id=market.id,
            copied_market_id=market_address,
            original_market_title=original_market_question,
            copied_market_title=market.question,
        )
        replicated_market_table_handler.save_replicated_markets([replicated_market])

        logger.info(
            f"Created `{created_market.url}` for `{market.question}` in category {category} out of {market.url}."
        )

        generate_and_set_image_for_market(
            market_address,
            market.question,
            api_keys,
        )

    return created_addresses


def omen_unfund_replicated_known_markets_tx(
    api_keys: APIKeys,
    saturation_above_threshold: float | None = None,
) -> None:
    from_address = api_keys.bet_from_address

    now = utcnow()
    # We want to unfund markets ~1 day before the resolution should be known.
    # That is, if the original market would be closing now, but we added `EXTEND_CLOSING_TIME_DELTA` to it,
    # we want to unfund any market that closes sooner than NOW + `EXTEND_CLOSING_TIME_DELTA` - 1 day.
    opened_before = now + EXTEND_CLOSING_TIME_DELTA - timedelta(days=1)

    # Fetch markets that we created, are soon to be known,
    # and still have liquidity in them (we didn't withdraw it yet).
    markets = OmenSubgraphHandler().get_omen_markets(
        limit=None,
        creator=from_address,
        question_opened_before=opened_before,
        liquidity_bigger_than=Wei(0),
    )

    for idx, market in enumerate(markets):
        # Optionally, if `saturation_above_threshold` is provided, skip markets that are not saturated to leave some free money motivation for agents.
        if (
            saturation_above_threshold is not None
            and market.is_open
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
            api_keys=api_keys,
            market=OmenAgentMarket.from_data_model(market),
            shares=None,
        )

    logger.info("Redeeming funds from unfunded markets.")
    redeem_from_all_user_positions(api_keys)
