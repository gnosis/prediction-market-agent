import os
import tempfile
import typing as t
from contextlib import contextmanager
from enum import Enum

from mech_client.interact import ConfirmationType, interact
from prediction_market_agent_tooling.gtypes import Probability
from pydantic import BaseModel

from prediction_market_agent.tools.mech.api_keys import MechAPIKeys
from prediction_market_agent.tools.mech.mech.packages.napthaai.customs.prediction_request_rag import (
    prediction_request_rag,
)
from prediction_market_agent.tools.mech.mech.packages.napthaai.customs.prediction_request_reasoning import (
    prediction_request_reasoning,
)
from prediction_market_agent.tools.mech.mech.packages.napthaai.customs.prediction_url_cot import (
    prediction_url_cot,
)
from prediction_market_agent.tools.mech.mech.packages.nickcom007.customs.prediction_request_sme import (
    prediction_request_sme,
)
from prediction_market_agent.tools.mech.mech.packages.polywrap.customs.prediction_with_research_report import (
    prediction_with_research_report,
)
from prediction_market_agent.tools.mech.mech.packages.valory.customs.prediction_request import (
    prediction_request,
)
from prediction_market_agent.utils import completion_str_to_json


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


def mech_request(question: str, mech_tool: MechTool) -> MechResponse:
    private_key = MechAPIKeys().bet_from_private_key.get_secret_value()
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
        return MechResponse.model_validate_json(response["result"])


def mech_request_local(
    question: str,
    mech_tool: MechTool,
) -> MechResponse:
    keys = MechAPIKeys()
    if mech_tool in [
        MechTool.PREDICTION_WITH_RESEARCH_REPORT,
        MechTool.PREDICTION_WITH_RESEARCH_REPORT_BOLD,
    ]:
        response = prediction_with_research_report.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "tavily": keys.tavily_api_key.get_secret_value(),
            },
        )
    elif mech_tool in [MechTool.PREDICTION_ONLINE, MechTool.PREDICTION_OFFLINE]:
        response = prediction_request.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "google_api_key": keys.google_search_api_key.get_secret_value(),
                "google_engine_id": keys.google_search_engine_id.get_secret_value(),
            },
        )
    elif mech_tool in [MechTool.PREDICTION_ONLINE_SME, MechTool.PREDICTION_OFFLINE_SME]:
        response = prediction_request_sme.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "google_api_key": keys.google_search_api_key.get_secret_value(),
                "google_engine_id": keys.google_search_engine_id.get_secret_value(),
            },
        )
    elif mech_tool == MechTool.PREDICTION_REQUEST_RAG:
        response = prediction_request_rag.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "google_api_key": keys.google_search_api_key.get_secret_value(),
                "google_engine_id": keys.google_search_engine_id.get_secret_value(),
            },
        )
    elif mech_tool == MechTool.PREDICTION_REQUEST_REASONING:
        response = prediction_request_reasoning.run(
            tool=mech_tool.value,
            prompt=question,
            api_keys={
                "openai": keys.openai_api_key.get_secret_value(),
                "google_api_key": keys.google_search_api_key.get_secret_value(),
                "google_engine_id": keys.google_search_engine_id.get_secret_value(),
            },
        )
    elif mech_tool == MechTool.PREDICTION_URL_COT:
        response = prediction_url_cot.run(
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
    return MechResponse.model_validate(result)
