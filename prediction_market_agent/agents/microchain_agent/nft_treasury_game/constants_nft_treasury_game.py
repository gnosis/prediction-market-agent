from prediction_market_agent_tooling.gtypes import xDai
from web3 import Web3

TREASURY_ADDRESS = Web3.to_checksum_address(
    "0x624ad0db52e6b18afb4d36b8e79d0c2a74f3fc8a"
)
NFT_TOKEN_FACTORY = Web3.to_checksum_address(
    "0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"
)

STARTING_AGENT_BALANCE = xDai(200)
STARTING_TREASURY_BALANCE = xDai(100)
# If someone withdraws something from the treasury, we consider it the end of the game.
TREASURY_THRESHOLD_BALANCE_TO_END_GAME = xDai(STARTING_TREASURY_BALANCE * 0.9)
