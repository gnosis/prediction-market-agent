from datetime import timedelta

from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.gtypes import HexAddress, HexBytes
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.data_models import RealityQuestion
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
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel


class FinalizeAndResolveResult(BaseModel):
    finalized: list[HexAddress]
    resolved: list[HexAddress]
    claimed_question_ids: list[HexBytes]


def omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx(
    private_credentials: PrivateCredentials,
) -> FinalizeAndResolveResult:
    public_key = private_credentials.public_key
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
        private_credentials,
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
        private_credentials,
        created_finalized_markets,
    )
    balances_after_resolution = get_balances(public_key)
    logger.info(f"{balances_after_resolution=}")

    # Fetch questions that are already finalised (last answer is older than 24 hours), but we didn't claim the bonded xDai yet.
    created_not_claimed_questions: list[
        RealityQuestion
    ] = OmenSubgraphHandler().get_questions(
        user=public_key,
        claimed=False,
        current_answer_before=before - timedelta(hours=24),
    )
    claimed_question_ids = claim_bonds_on_realitio_questions(
        private_credentials,
        created_not_claimed_questions,
        auto_withdraw=True,
    )
    balances_after_claiming = get_balances(public_key)
    logger.info(f"{balances_after_claiming=}")

    return FinalizeAndResolveResult(
        finalized=finalized_markets,
        resolved=resolved_markets,
        claimed_question_ids=claimed_question_ids,
    )
