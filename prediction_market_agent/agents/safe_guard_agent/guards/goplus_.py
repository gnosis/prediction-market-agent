import time
from typing import Callable, TypeVar

import tenacity
from goplus.address import Address
from goplus.nft import Nft
from goplus.token import Token
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)


class GoPlusError(Exception):
    pass


@observe()
def validate_safe_transaction_goplus_token_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult | None:
    addr = new_transaction_safetx.to
    data = _goplus_call(
        lambda: Token().token_security(
            chain_id=f"{RPCConfig().chain_id}", addresses=[addr]
        )
    )
    if data is None or not data.result:
        return None

    logger.info(data.result)
    result_for_addr = data.result[addr.lower()]
    malicious_reasons: list[str] = []

    if _goplus_to_bool(result_for_addr.anti_whale_modifiable):
        malicious_reasons.append("Token is anti-whale modifiable.")

    if _goplus_to_bool(result_for_addr.buy_tax):
        malicious_reasons.append("Token has buy tax.")

    if _goplus_to_bool(result_for_addr.can_take_back_ownership):
        malicious_reasons.append("Token can take back ownership.")

    if _goplus_to_bool(result_for_addr.external_call):
        malicious_reasons.append("Token has external calls implemented.")

    if _goplus_to_bool(result_for_addr.fake_token):
        malicious_reasons.append("Token is fake.")

    if _goplus_to_bool(result_for_addr.hidden_owner):
        malicious_reasons.append("Token has hidden owner.")

    if _goplus_to_bool(result_for_addr.is_airdrop_scam):
        malicious_reasons.append("Token is an airdrop scam.")

    if _goplus_to_bool(result_for_addr.is_anti_whale):
        malicious_reasons.append("Token is anti-whale.")

    if _goplus_to_bool(result_for_addr.is_blacklisted):
        malicious_reasons.append("Token is blacklisted.")

    if _goplus_to_bool(result_for_addr.is_honeypot):
        malicious_reasons.append("Token is a honeypot.")

    if not _goplus_to_bool(result_for_addr.is_open_source):
        malicious_reasons.append("Token contract is not open source.")

    # Direct check for `False`, because None is fine here.
    if _goplus_to_bool(result_for_addr.is_true_token) is False:
        malicious_reasons.append("Token is not a true token.")

    if _goplus_to_bool(result_for_addr.is_whitelisted):
        malicious_reasons.append("Token is whitelisted.")

    if other_potential_risks := result_for_addr.other_potential_risks:
        malicious_reasons.append(
            f"Token has other potential risks ({other_potential_risks})."
        )

    if _goplus_to_bool(result_for_addr.selfdestruct):
        malicious_reasons.append("Token has self-destruct function.")

    return _build_validation_result(malicious_reasons)


@observe()
def validate_safe_transaction_goplus_address_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult | None:
    addr = new_transaction_safetx.to
    data = _goplus_call(
        lambda: Address().address_security(
            chain_id=f"{RPCConfig().chain_id}", address=addr
        )
    )
    if data is None:
        return None

    logger.info(data.result)
    malicious_reasons: list[str] = []

    if _goplus_to_bool(data.result.blacklist_doubt):
        malicious_reasons.append("Address is blacklisted.")

    if _goplus_to_bool(data.result.blackmail_activities):
        malicious_reasons.append("Address has reported blackmail activities.")

    if _goplus_to_bool(data.result.cybercrime):
        malicious_reasons.append("Address has reported cybercrime activities.")

    if _goplus_to_bool(data.result.darkweb_transactions):
        malicious_reasons.append("Address has darkweb transactions.")

    if _goplus_to_bool(data.result.fake_kyc):
        malicious_reasons.append("Address has fake KYC.")

    if _goplus_to_bool(data.result.fake_standard_interface):
        malicious_reasons.append(
            "Address has fake standard interface (usually happens with scam assets)."
        )

    if _goplus_to_bool(data.result.fake_token):
        malicious_reasons.append("Address has fake token.")

    if _goplus_to_bool(data.result.financial_crime):
        malicious_reasons.append("Address has financial crime activities.")

    if _goplus_to_bool(data.result.gas_abuse):
        malicious_reasons.append("Address has gas abuse activities.")

    if _goplus_to_bool(data.result.honeypot_related_address):
        malicious_reasons.append("Address is related to a honeypot.")

    if _goplus_to_bool(data.result.malicious_mining_activities):
        malicious_reasons.append("Address has malicious mining activities.")

    if _goplus_to_bool(data.result.mixer):
        malicious_reasons.append("Address is a mixer.")

    if _goplus_to_bool(data.result.money_laundering):
        malicious_reasons.append("Address has money laundering activities.")

    if _goplus_to_bool(data.result.number_of_malicious_contracts_created):
        malicious_reasons.append("Address has created a number of malicious contracts.")

    if _goplus_to_bool(data.result.phishing_activities):
        malicious_reasons.append("Address has reported phishing activities.")

    if _goplus_to_bool(data.result.reinit):
        malicious_reasons.append("Address could be redeployed with a different code.")

    if _goplus_to_bool(data.result.sanctioned):
        malicious_reasons.append("Address is sanctioned.")

    if _goplus_to_bool(data.result.stealing_attack):
        malicious_reasons.append("Address has stealing attack activities.")

    return _build_validation_result(malicious_reasons)


@observe()
def validate_safe_transaction_goplus_nft_security(
    new_transaction: DetailedTransactionResponse,
    new_transaction_safetx: SafeTx,
    history: list[DetailedTransactionResponse],
) -> ValidationResult | None:
    addr = new_transaction_safetx.to
    data = _goplus_call(
        lambda: Nft().nft_security(chain_id=f"{RPCConfig().chain_id}", address=addr)
    )
    if data is None:
        return None

    logger.info(data.result)
    malicious_reasons: list[str] = []

    if _goplus_to_bool(data.result.malicious_nft_contract):
        malicious_reasons.append("Malicious NFT activity reported.")

    if not _goplus_to_bool(data.result.nft_open_source):
        malicious_reasons.append("NFT contract is not open source.")

    if _goplus_to_bool(data.result.privileged_burn.value):
        malicious_reasons.append("NFT contract has privileged burn function.")

    if _goplus_to_bool(data.result.self_destruct.value):
        malicious_reasons.append("NFT contract has self-destruct function.")

    if _goplus_to_bool(data.result.transfer_without_approval.value):
        malicious_reasons.append("NFT contract has transfer without approval function.")

    return _build_validation_result(malicious_reasons)


def _build_validation_result(malicious_reasons: list[str]) -> ValidationResult:
    return ValidationResult(
        ok=not malicious_reasons,
        reason=" ".join(malicious_reasons) if malicious_reasons else "No issues found.",
    )


T = TypeVar("T")


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_not_exception_type(GoPlusError),
)
def _goplus_call(f: Callable[[], T], retry: bool = True) -> T | None:
    response = f()
    code = int(response.code)  # type: ignore # GoPlus isn't typed.

    # See https://docs.gopluslabs.io/reference/api-status-code for possible status codes.
    if code == 1:
        return response

    elif code in (2, 4029) and retry:
        logger.warning(f"Goplus rate limit hit, waiting a bit and trying again.")
        time.sleep(16)  # Sleep time based on the doc above.
        return _goplus_call(f, retry=False)

    elif code == 2 and not retry:
        # If we don't have full data even after retry, let's use them anyway.
        logger.warning(f"Goplus returns partial data even after retry.")
        return response

    elif code in (2020, 2021):
        # Expected errors, like trying to verify non-contract address.
        return None

    raise GoPlusError(f"Goplus error: {response}")


def _goplus_to_bool(v: str | int | float | bool | None) -> bool | None:
    """
    GoPlus often returns "0", "1", "2", etc. or None in the responses.
    Use this as a common function to parse them into simple iffable response.
    """
    return bool(float(v)) if v is not None and str(v).strip() != "" else None
