from prediction_market_agent.utils import APIKeys


class MicrochainAgentKeys(APIKeys):
    # Double check to make sure you want to actually post on public social media.
    ENABLE_SOCIAL_MEDIA: bool = False
