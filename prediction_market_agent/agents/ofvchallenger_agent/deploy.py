from datetime import timedelta
from functools import partial

import langfuse
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import ChecksumAddress, HexBytes, xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.data_models import (
    OmenMarket,
    RealityResponse,
)
from prediction_market_agent_tooling.markets.omen.omen_contracts import MetriSuperGroup
from prediction_market_agent_tooling.markets.omen.omen_resolving import (
    Resolution,
    omen_submit_answer_market_tx,
    omen_submit_invalid_answer_market_tx,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
    RealityResponse,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.omen.reality_accuracy import reality_accuracy
from prediction_market_agent_tooling.tools.utils import retry_until_true, utcnow
from pydantic import BaseModel
from web3 import Web3

from prediction_market_agent.agents.ofvchallenger_agent.ofv_resolver import (
    ofv_answer_binary_question,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    claim_all_bonds_on_reality,
)
from prediction_market_agent.agents.specialized_agent.deploy import (
    SPECIALIZED_FOR_MARKET_CREATORS,
)
from prediction_market_agent.agents.utils import (
    build_resolution_from_factuality_for_omen_market,
)
from prediction_market_agent.utils import APIKeys

OFV_CHALLENGER_TAG = "ofv_challenger"
OFV_CHALLENGER_EOA_ADDRESS = Web3.to_checksum_address(
    "0xA01143e55e014E874c8f59B5e8EFB5CEe9B8cdBF"
)
OFV_CHALLENGER_SAFE_ADDRESS = Web3.to_checksum_address(
    "0x9D0260500ba7b068b5b0f4AfA9F8864eBc0B059a"
)
NORMAL_CHALLENGE_BOND = xDai(10)
SMALL_CHALLENGE_BOND = xDai(0.01)
INFINITE_GAMES_MARKET_CREATOR = Web3.to_checksum_address(
    "0xecfd8717deada8ccfb925a3f7ea31d47cbb4f37f"
)
# We benchmarked OFV only against Olas market-creators.
MARKET_CREATORS_TO_CHALLENGE: list[ChecksumAddress] | None = [
    # Olas market-creator 0.
    Web3.to_checksum_address("0x89c5cc945dd550bcffb72fe42bff002429f46fec"),
    # Olas market-creator 1.
    Web3.to_checksum_address("0xffc8029154ecd55abed15bd428ba596e7d23f557"),
    INFINITE_GAMES_MARKET_CREATOR,
    # Some old market creator.
    Web3.to_checksum_address("0x92f869018b5f954a4197a15feb951cf9260c54a8"),
] + (
    # But also use it to challenge the specialized markets (e.g. DevConflict), as we don't have anything better.
    SPECIALIZED_FOR_MARKET_CREATORS
)
COLLATERAL_TOKENS_TO_CHALLENGE_FROM_ANY_MARKET_CREATOR: list[ChecksumAddress] | None = [
    # We challenge any market creators that use Metri/Circles, because we love Circles!
    MetriSuperGroup().address,
]


class Challenge(BaseModel):
    old_responses: list[RealityResponse]
    new_resolution: Resolution | None
    reasoning: str


class OFVChallengerAgent(DeployableAgent):
    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError("Can challenge only Omen.")

        api_keys = APIKeys()
        self.challenge(api_keys)

    @observe()
    def challenge(self, api_keys: APIKeys) -> None:
        self.langfuse_update_current_trace(tags=[OFV_CHALLENGER_TAG])

        # Claim the bonds as first thing, to have funds for the new challenges.
        claim_all_bonds_on_reality(api_keys)

        get_omen_binary_markets_common_filters_with_limit_and_question_opened = partial(
            OmenSubgraphHandler().get_omen_markets,
            limit=None,
            # We need markets already opened for answers.
            question_opened_before=utcnow(),
            # OFVChallenger can handle only Yes/No markets.
            include_categorical_markets=False,
            include_scalar_markets=False,
        )
        markets_open_for_answers: list[OmenMarket] = []

        if MARKET_CREATORS_TO_CHALLENGE is not None:
            get_omen_binary_markets_common_filters_with_market_creators = partial(
                get_omen_binary_markets_common_filters_with_limit_and_question_opened,
                creator_in=MARKET_CREATORS_TO_CHALLENGE,
            )
            markets_open_for_answers += get_omen_binary_markets_common_filters_with_market_creators(
                # With a little bandwidth for the market to be finalized,
                # so we have time for processing it without erroring out at the end.
                question_finalized_after=utcnow()
                + timedelta(minutes=30),
            ) + get_omen_binary_markets_common_filters_with_market_creators(
                # And also markets without any answer at all yet.
                question_with_answers=False,
            )
        if COLLATERAL_TOKENS_TO_CHALLENGE_FROM_ANY_MARKET_CREATOR is not None:
            markets_open_for_answers += get_omen_binary_markets_common_filters_with_limit_and_question_opened(
                # With a little bandwidth for the market to be finalized,
                # so we have time for processing it without erroring out at the end.
                question_finalized_after=utcnow() + timedelta(minutes=30),
                collateral_token_address_in=tuple(
                    COLLATERAL_TOKENS_TO_CHALLENGE_FROM_ANY_MARKET_CREATOR
                ),
            ) + get_omen_binary_markets_common_filters_with_limit_and_question_opened(
                # And also markets without any answer at all yet.
                question_with_answers=False,
                collateral_token_address_in=tuple(
                    COLLATERAL_TOKENS_TO_CHALLENGE_FROM_ANY_MARKET_CREATOR
                ),
            )

        logger.info(f"Found {len(markets_open_for_answers)} markets to challenge.")

        for market in markets_open_for_answers:
            if (
                market.creator_checksum == INFINITE_GAMES_MARKET_CREATOR
                and market.question.answerFinalizedTimestamp is None
                and (utcnow() - market.close_time) < timedelta(days=3)
            ):
                logger.info(
                    f"Skipping resolution of {market.url=} for now, to give them time to post the first answer."
                )
                continue

            self.challenge_market(market, api_keys)

        # Compute accuracy on Reality and report as error if it goes down too much.
        last_week_accuracy = reality_accuracy(
            api_keys.bet_from_address, timedelta(days=7)
        )
        (logger.info if last_week_accuracy.accuracy >= 0.8 else logger.error)(
            f"Last weeks accuracy is {last_week_accuracy.accuracy} on {last_week_accuracy.total} questions."
        )

    @retry_until_true(
        # We have a bug where subgraph sometimes return empty list of responses, even though there already are some.
        # Try to retry fetching responses multiple time, to see if some appear.
        lambda responses: len(responses)
        > 0
    )
    def get_responses_until_available(
        self,
        question_id: HexBytes,
    ) -> list[RealityResponse]:
        return OmenSubgraphHandler().get_responses(limit=None, question_id=question_id)

    @observe()
    def challenge_market(
        self,
        market: OmenMarket,
        api_keys: APIKeys,
        web3: Web3 | None = None,
    ) -> Challenge:
        logger.info(f"Challenging market {market.url=}")
        langfuse.get_client().update_current_trace(metadata={"url": market.url})

        # Use just a small bond on questions that are either finalized too quickly (might not receive corrections in time),
        # or finalized in too much in the future (not worth locking significant funds there).
        if (
            timedelta(hours=23)
            <= market.question.timeout_timedelta
            <= timedelta(days=7)
        ):
            bond = NORMAL_CHALLENGE_BOND
        else:
            bond = SMALL_CHALLENGE_BOND

        logger.info(
            f"Using {bond=} for {market.url=} with timeout {market.question.timeout_timedelta=}."
        )

        existing_responses = self.get_responses_until_available(
            question_id=market.question.question_id
        )
        logger.info(
            f"{market.url=}'s responses and bonds: {[(r.answer, r.bond_xdai) for r in existing_responses]}"
        )

        # We don't plan to re-challenge markets already challenged by the challenger, should we?
        if any(
            response.user_checksummed == api_keys.bet_from_address
            for response in existing_responses
        ):
            logger.info(
                f"Market {market.url=} already challenged by challenger. Skipping."
            )
            return Challenge(
                old_responses=existing_responses,
                new_resolution=None,
                reasoning=f"Already challenged by {api_keys.bet_from_address=}.",
            )

        # Next bond needs to be at least double the previous one.
        if any(response.bond_xdai >= bond / 2 for response in existing_responses):
            logger.info(
                f"Market {market.url=} already challenged with bond > {bond} / 2. Skipping."
            )
            return Challenge(
                old_responses=existing_responses,
                new_resolution=None,
                reasoning=f"Already challenged with bond > {bond} / 2.",
            )

        try:
            answer = ofv_answer_binary_question(market.question_title, api_keys)
        except Exception as e:
            logger.exception(
                f"Exception while getting factuality for market {market.url=}. Skipping. Exception: {e}"
            )
            return Challenge(
                old_responses=existing_responses,
                new_resolution=None,
                reasoning=f"Exception in OFV: {str(e)}",
            )

        if answer is None:
            logger.warning(
                f"OFV didn't factcheck {market.url=}, question {market.question_title=}. Skipping."
            )
            return Challenge(
                old_responses=existing_responses,
                new_resolution=None,
                reasoning="OFV failed to provide an answer.",
            )

        new_resolution = build_resolution_from_factuality_for_omen_market(
            factuality=answer.factuality
        )

        logger.info(
            f"Challenging market {market.url=} with resolution {new_resolution=}"
        )

        if not new_resolution.invalid:
            omen_submit_answer_market_tx(
                api_keys=api_keys,
                market=market,
                resolution=new_resolution,
                bond=bond,
                web3=web3,
            )
        else:
            omen_submit_invalid_answer_market_tx(
                api_keys=api_keys, market=market, bond=bond, web3=web3
            )

        return Challenge(
            old_responses=existing_responses,
            new_resolution=new_resolution,
            reasoning="Challenge response submitted.",
        )
