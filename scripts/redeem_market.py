from web3 import Web3

from prediction_market_agent.omen import get_market, omen_claim_winnings
from prediction_market_agent.tools.gnosis_rpc import GNOSIS_RPC_URL
from prediction_market_agent.tools.web3_utils import check_tx_receipt
from prediction_market_agent.utils import get_bet_from_address, get_bet_from_private_key

web3 = Web3(Web3.HTTPProvider(GNOSIS_RPC_URL))

market_address = "0x1a875b15564939d640ad1ef13769eab5ec74ef03"

market = get_market(market_address)
claim_receipt = omen_claim_winnings(
    web3=web3,
    market=market,
    from_address=get_bet_from_address(),
    from_private_key=get_bet_from_private_key(),
)
check_tx_receipt(claim_receipt)

"""
TODO debug error web3.eth.estimate_gas ValueError: {'code': -32015, 'message': 'revert'}

>>> transaction
{'nonce': 10, 'from': '0x3666DA333dAdD05083FEf9FF6dDEe588d26E4307', 'to': '0x1A875b15564939d640aD1Ef13769EAB5eC74Ef03', 'data': '0x01b7037c000000000000000000000000e91d153e0b41518a2ce8dd3d7944fa863463a97d00000000000000000000000000000000000000000000000000000000000000006612c884ca0fa102b441733774fd7fe31f4a8a5991adc3dcdce980d2f28520c00000000000000000000000000000000000000000000000000000000000000080000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000002'}
"""
