from typing import Annotated

from prediction_market_agent_tooling.gtypes import xDai, xdai_type
from prediction_market_agent_tooling.loggers import logger
from pydantic import BeforeValidator

from prediction_market_agent.utils import APIKeys


def xdai_none_validator(value: str | float) -> xDai | None:
    if str(value).lower().strip() == "none":
        return None
    return xdai_type(value)


class MicrochainAgentKeys(APIKeys):
    # Double check to make sure you want to actually post on public social media.
    ENABLE_SOCIAL_MEDIA: bool = False
    # Double check to not spend big money during testing.
    SENDING_XDAI_CAP: Annotated[xDai | None, BeforeValidator(xdai_none_validator)] = (
        xdai_type(0.1)
    )
    # Double check to not transfer NFTs during testing.
    ENABLE_NFT_TRANSFER: bool = False

    def cap_sending_xdai(self, amount: xDai) -> xDai:
        if self.SENDING_XDAI_CAP is None:
            return amount
        amount = xDai(min(amount, self.SENDING_XDAI_CAP))
        logger.warning(f"Caping sending xDai value to {amount}.")
        return amount
