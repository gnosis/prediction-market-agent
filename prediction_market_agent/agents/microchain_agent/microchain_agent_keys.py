from typing import Annotated

from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.loggers import logger
from pydantic import BeforeValidator

from prediction_market_agent.utils import APIKeys


def xdai_none_validator(value: str | float | None) -> xDai | None:
    if value is None or str(value).lower().strip() == "none":
        return None
    return xDai(value)


class MicrochainAgentKeys(APIKeys):
    # Double check to make sure you want to actually post on public social media.
    ENABLE_SOCIAL_MEDIA: bool = False
    # Double check to not spend big money during testing.
    SENDING_XDAI_CAP: Annotated[
        xDai | None, BeforeValidator(xdai_none_validator)
    ] = xDai(0.1)
    # Double check to not transfer NFTs during testing.
    ENABLE_NFT_TRANSFER: bool = False

    def cap_sending_xdai(self, amount: xDai) -> xDai:
        if self.SENDING_XDAI_CAP is None:
            return amount
        amount = min(amount, self.SENDING_XDAI_CAP)
        logger.warning(f"Caping sending xDai value to {amount}.")
        return amount
