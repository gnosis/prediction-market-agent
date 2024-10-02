from datetime import timedelta

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import HexAddress, HexBytes, xDai, xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.data_models import (
    RealityQuestion,
    RealityResponse,
)
from prediction_market_agent_tooling.markets.omen.omen_resolving import (
    claim_bonds_on_realitio_questions,
    finalize_markets,
    find_resolution_on_other_markets,
    resolve_markets,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import DatetimeUTC, utcnow
from pydantic import BaseModel


class ClaimResult(BaseModel):
    claimed_question_ids: list[HexBytes]
    claimed: xDai


class FinalizeAndResolveResult(BaseModel):
    finalized: list[HexAddress]
    resolved: list[HexAddress]
    claimed: ClaimResult


@observe()
def omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx(
    api_keys: APIKeys,
) -> FinalizeAndResolveResult:
    public_key = api_keys.bet_from_address
    balances_start = get_balances(public_key)
    logger.info(f"{balances_start=}")

    # Just to be friendly with timezones.
    before = utcnow() - timedelta(hours=8)

    # Fetch markets created by us that are already open, but no answer was submitted yet.
    created_opened_markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=public_key,
        opened_before=before,
        finalized=False,
    )
    # Finalize them (set answer on Realitio).
    created_opened_markets_with_resolutions = [
        (m, find_resolution_on_other_markets(m)) for m in created_opened_markets
    ]
    finalized_markets = finalize_markets(
        api_keys,
        created_opened_markets_with_resolutions,
    )
    balances_after_finalization = get_balances(public_key)
    logger.info(f"{balances_after_finalization=}")

    # Fetch markets created by us that are already open, and we already submitted an answer more than a day ago, but they aren't resolved yet.
    created_finalized_markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=public_key,
        finalized_before=before - timedelta(hours=24),
        resolved=False,
    )
    # Resolve them (resolve them on Oracle).
    resolved_markets = resolve_markets(
        api_keys,
        created_finalized_markets,
    )
    balances_after_resolution = get_balances(public_key)
    logger.info(f"{balances_after_resolution=}")

    claimed = claim_all_bonds_on_reality(api_keys, finalized_before=before)

    return FinalizeAndResolveResult(
        finalized=finalized_markets,
        resolved=resolved_markets,
        claimed=claimed,
    )


@observe()
def claim_all_bonds_on_reality(
    api_keys: APIKeys, finalized_before: DatetimeUTC | None = None
) -> ClaimResult:
    # Just to be friendly with time differences.
    finalized_before = finalized_before or utcnow()
    public_key = api_keys.bet_from_address

    balances_before_claiming = get_balances(public_key)
    logger.info(f"{balances_before_claiming=}")

    # Fetch our responses that are on already finalised questions, but we didn't claim the bonded xDai yet.
    responses: list[RealityResponse] = OmenSubgraphHandler().get_responses(
        limit=None,
        user=public_key,
        question_claimed=False,
        question_finalized_before=finalized_before,
    )
    # Extract only the unique questions out of responses (there could be multiple responses for the same question, but only the whole question can be claimed once).
    questions: list[RealityQuestion] = list(
        {r.question.questionId: r.question for r in responses}.values()
    )
    claimed_question_ids = claim_bonds_on_realitio_questions(
        api_keys,
        questions,
        auto_withdraw=True,
    )
    balances_after_claiming = get_balances(public_key)
    logger.info(f"{balances_after_claiming=}")

    return ClaimResult(
        claimed_question_ids=claimed_question_ids,
        claimed=xdai_type(balances_after_claiming.xdai - balances_before_claiming.xdai),
    )
