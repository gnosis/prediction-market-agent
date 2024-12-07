from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen import OMEN_TINY_BET_AMOUNT

from prediction_market_agent.utils import APIKeys


class MicrochainAgentKeys(APIKeys):
    # Double check to make sure you want to actually post on public social media.
    ENABLE_SOCIAL_MEDIA: bool = False
    # Double check to not spend big money during testing.
    SENDING_XDAI_CAP: float | None = OMEN_TINY_BET_AMOUNT
    # Double check to not transfer NFTs during testing.
    ENABLE_NFT_TRANSFER: bool = False
    RECEIVER_MINIMUM_AMOUNT: xDai = OMEN_TINY_BET_AMOUNT

    def cap_sending_xdai(self, amount: xDai) -> xDai:
        if self.SENDING_XDAI_CAP is None:
            return amount
        amount = xDai(min(amount, self.SENDING_XDAI_CAP))
        logger.warning(f"Caping sending xDai value to {amount}.")
        return amount
