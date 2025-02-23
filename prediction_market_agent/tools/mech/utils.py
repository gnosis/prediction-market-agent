import os
import tempfile
import typing as t
from contextlib import contextmanager
from enum import Enum
from importlib.util import find_spec

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


def is_package_available(package_name: str) -> bool:
    """
    Asserts whether the specified package is available.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        True if the package is available, False otherwise.
    """
    if find_spec(package_name):
        return True
    return False


def mech_request(question: str, mech_tool: MechTool) -> MechResponse:
    if not is_package_available("mech_client"):
        raise EnvironmentError("Package `mech_client` is not installed.")

    from mech_client.interact import ConfirmationType, interact

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
            agent_id=6,
            private_key_path=tmpfile_path,
            # To see a list of available tools, comment out the tool parameter
            # and run the function. You will be prompted to select a tool.
            tool=mech_tool.value,
            confirmation_type=ConfirmationType.WAIT_FOR_BOTH,
        )
        del os.environ[mech_strategy_env_var]
        return MechResponse.model_validate_json(response["result"])
