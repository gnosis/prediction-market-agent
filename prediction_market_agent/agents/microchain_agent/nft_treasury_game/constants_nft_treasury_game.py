from prediction_market_agent_tooling.gtypes import xDai

STARTING_AGENT_BALANCE = xDai(200)
STARTING_TREASURY_BALANCE = xDai(100)
# If someone withdraws something from the treasury, we consider it the end of the game.
TREASURY_THRESHOLD_BALANCE_TO_END_GAME = xDai(STARTING_TREASURY_BALANCE * 0.9)
