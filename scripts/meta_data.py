from prediction_market_agent_tooling.markets.markets import get_binary_markets
from prediction_market_agent_tooling.markets.market_type import MarketType

markets = get_binary_markets(limit=3, market_type=MarketType.OMEN)

m = markets[0]

print(type(m))
print("\nDIR:\n", dir(m))

if hasattr(m, "model_dump"):
    md = m.model_dump()
    print("\nMODEL_DUMP KEYS:\n", md.keys())
    print("\nFULL MODEL_DUMP:\n", md)
else:
    print("\nDICT:\n", m.__dict__)
