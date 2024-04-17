import os
import tempfile
import typing as t
from contextlib import contextmanager
from decimal import Decimal
from enum import Enum

from mech_client.interact import ConfirmationType, interact
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.data_models import BetAmount
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.data_models import (
    OMEN_FALSE_OUTCOME,
    OMEN_TRUE_OUTCOME,
)
from prediction_market_agent_tooling.markets.omen.data_models import (
    get_boolean_outcome as get_omen_boolean_outcome,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel, SecretStr

from prediction_market_agent.agents.microchain_agent.mech.mech.packages.polywrap.customs.prediction_with_research_report import (
    prediction_with_research_report,
)
from prediction_market_agent.agents.microchain_agent.mech.mech.packages.valory.customs.prediction_request import (
    prediction_request,
)
from prediction_market_agent.utils import APIKeys, completion_str_to_json


class MicrochainAPIKeys(APIKeys):
    GOOGLE_SEARCH_API_KEY: t.Optional[SecretStr] = None
    GOOGLE_SEARCH_ENGINE_ID: t.Optional[SecretStr] = None

    @property
    def google_search_api_key(self) -> SecretStr:
        return check_not_none(
            self.GOOGLE_SEARCH_API_KEY,
            "GOOGLE_SEARCH_API_KEY missing in the environment.",
        )

    @property
    def google_search_engine_id(self) -> SecretStr:
        return check_not_none(
            self.GOOGLE_SEARCH_ENGINE_ID,
            "GOOGLE_SEARCH_ENGINE_ID missing in the environment.",
        )


class MicroMarket(BaseModel):
    question: str
    id: str

    @staticmethod
    def from_agent_market(market: AgentMarket) -> "MicroMarket":
        return MicroMarket(
            question=market.question,
            id=market.id,
        )

    def __str__(self) -> str:
        return f"'{self.question}', id: {self.id}"


class MechResult(BaseModel):
    p_yes: float
    p_no: float
    confidence: float
    info_utility: float


def get_binary_markets(market_type: MarketType) -> list[AgentMarket]:
    # Get the 5 markets that are closing soonest
    cls = market_type.market_class
    markets: t.Sequence[AgentMarket] = cls.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=(
            SortBy.NONE
            if market_type == MarketType.POLYMARKET
            else SortBy.CLOSING_SOONEST
        ),
        limit=5,
    )
    return list(markets)


def get_balance(market_type: MarketType) -> BetAmount:
    currency = market_type.market_class.currency
    if market_type == MarketType.OMEN:
        # We focus solely on xDAI balance for now to avoid the agent having to wrap/unwrap xDAI.
        return BetAmount(
            amount=Decimal(get_balances(MicrochainAPIKeys().bet_from_address).xdai),
            currency=currency,
        )
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_boolean_outcome(market_type: MarketType, outcome: str) -> bool:
    if market_type == MarketType.OMEN:
        return get_omen_boolean_outcome(outcome)
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_yes_outcome(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return OMEN_TRUE_OUTCOME
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_no_outcome(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return OMEN_FALSE_OUTCOME
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


def get_example_market_id(market_type: MarketType) -> str:
    if market_type == MarketType.OMEN:
        return "0x0020d13c89140b47e10db54cbd53852b90bc1391"
    else:
        raise ValueError(f"Market type '{market_type}' not supported")


@contextmanager
def saved_str_to_tmpfile(s: str) -> t.Iterator[str]:
    # Write the string to the temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(s.encode())

    yield tmp.name

    # Finally remove the temporary file
    os.remove(tmp.name)


class MechTool(str, Enum):
    PREDICTION_WITH_RESEARCH_REPORT = "prediction-with-research-report-conservative"
    PREDICTION_ONLINE = "prediction-online"


def mech_request(question: str, mech_tool: MechTool) -> MechResult:
    private_key = MicrochainAPIKeys().bet_from_private_key.get_secret_value()
    with saved_str_to_tmpfile(private_key) as tmpfile_path:
        # Increase gas price to reduce chance of 'out of gas' transaction failures
        mech_strategy_env_var = "MECHX_LEDGER_DEFAULT_GAS_PRICE_STRATEGY"
        if os.getenv(mech_strategy_env_var):
            raise ValueError(f"{mech_strategy_env_var} already set in the environment.")
        os.environ[mech_strategy_env_var] = "gas_station"

        response = interact(
            prompt=question,
            # Taken from https://github.com/valory-xyz/mech?tab=readme-ov-file#examples-of-deployed-mechs
            agent_id=6,
            private_key_path=tmpfile_path,
            # To see a list of available tools, comment out the tool parameter
            # and run the function. You will be prompted to select a tool.
            tool=mech_tool.value,
            confirmation_type=ConfirmationType.WAIT_FOR_BOTH,
        )
        del os.environ[mech_strategy_env_var]
        return MechResult.model_validate_json(response["result"])


def mech_request_local(question: str, mech_tool: MechTool) -> MechResult:
    keys = MicrochainAPIKeys()
    if mech_tool == MechTool.PREDICTION_WITH_RESEARCH_REPORT:
        response = prediction_with_research_report.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "tavily": keys.tavily_api_key.get_secret_value(),
            },
        )
    elif mech_tool == MechTool.PREDICTION_ONLINE:
        response = prediction_request.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "google_api_key": keys.google_search_api_key.get_secret_value(),
                "google_engine_id": keys.google_search_engine_id.get_secret_value(),
            },
        )
    else:
        raise ValueError(f"Mech type '{mech_tool}' not supported")

    result = completion_str_to_json(str(response[0]))
    return MechResult.model_validate(result)
