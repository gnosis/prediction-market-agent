import os
import tempfile
import typing as t
from contextlib import contextmanager
from enum import Enum

from mech_client.interact import ConfirmationType, interact
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from pydantic import BaseModel

from prediction_market_agent.utils import APIKeys


class MechResponse(BaseModel):
    p_yes: Probability
    p_no: Probability
    confidence: Probability
    info_utility: Probability


@contextmanager
def saved_str_to_tmpfile(s: str) -> t.Iterator[str]:
    # Write the string to the temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(s.encode())

    yield tmp.name

    # Finally remove the temporary file
    os.remove(tmp.name)


class MechTool(str, Enum):
    PREDICTION_WITH_RESEARCH_REPORT = "prediction-with-research-conservative"
    PREDICTION_WITH_RESEARCH_REPORT_BOLD = "prediction-with-research-bold"
    PREDICTION_ONLINE = "prediction-online"
    PREDICTION_OFFLINE = "prediction-offline"
    PREDICTION_ONLINE_SME = "prediction-online-sme"
    PREDICTION_OFFLINE_SME = "prediction-offline-sme"
    PREDICTION_REQUEST_RAG = "prediction-request-rag"
    PREDICTION_REQUEST_REASONING = "prediction-request-reasoning"
    PREDICTION_URL_COT = "prediction-url-cot"


def mech_request_using_mech_enum(question: str, mech_tool: MechTool) -> MechResponse:
    return mech_request(question, mech_tool.value)


def mech_request(question: str, mech_tool_id: str, agent_id: int) -> MechResponse:
    private_key = APIKeys().bet_from_private_key.get_secret_value()
    with saved_str_to_tmpfile(private_key) as tmpfile_path:
        # Increase gas price to reduce chance of 'out of gas' transaction failures
        mech_strategy_env_var = "MECHX_LEDGER_DEFAULT_GAS_PRICE_STRATEGY"
        if (
            os.getenv(mech_strategy_env_var)
            and os.getenv(mech_strategy_env_var) != "gas_station"
        ):
            logger.warning(f"Overwriting {mech_strategy_env_var} with 'gas_station'")
        os.environ[mech_strategy_env_var] = "gas_station"

        response = interact(
            prompt=question,
            # Taken from https://github.com/valory-xyz/mech?tab=readme-ov-file#examples-of-deployed-mechs
            agent_id=agent_id,
            private_key_path=tmpfile_path,
            tool=mech_tool_id,
            confirmation_type=ConfirmationType.OFF_CHAIN,
        )
        del os.environ[mech_strategy_env_var]
        # return MechResponse.model_validate_json(response["result"])
        return response
