import time
from datetime import timedelta
from typing import Any, Callable, TypeVar

import tenacity
from goplus.address import Address
from goplus.nft import Nft
from goplus.token import Token
from prediction_market_agent_tooling.config import RPCConfig
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.caches.db_cache import db_cache
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_guard_agent.guards.abstract_guard import (
    AbstractGuard,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.tools.streamlit_utils import dict_to_point_list


class GoPlusError(Exception):
    pass


class GoPlusTokenSecurity(AbstractGuard):
    name = "Token security"
    description = "This guard checks the security of tokens involved in the transaction using GoPlus API."

    @observe(name="validate_safe_transaction_goplus_token_security")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
    ) -> ValidationResult | None:
        results = {
            addr: res
            for addr in all_addresses_from_tx
            if (res := goplus_token_security(addr)) is not None
        }
        if not results:
            return None

        malicious_reasons: list[str] = []

        for addr in all_addresses_from_tx:
            result_for_addr = results.get(addr)
            if result_for_addr is None:
                continue

            logger.info(
                f"{self.name} response for address {addr}:\n\n{dict_to_point_list(result_for_addr)}",
                streamlit=True,
            )

            if _goplus_to_bool(result_for_addr["anti_whale_modifiable"]):
                malicious_reasons.append(f"Token {addr} is anti-whale modifiable.")

            if _goplus_to_bool(result_for_addr["buy_tax"]):
                malicious_reasons.append(f"Token {addr} has buy tax.")

            if _goplus_to_bool(result_for_addr["can_take_back_ownership"]):
                malicious_reasons.append(f"Token {addr} can take back ownership.")

            # Probably not malicious.
            # if _goplus_to_bool(result_for_addr["external_call"]):
            #     malicious_reasons.append(f"Token {addr} has external calls implemented.")

            if _goplus_to_bool(result_for_addr["fake_token"]):
                malicious_reasons.append(f"Token {addr} is fake.")

            if _goplus_to_bool(result_for_addr["hidden_owner"]):
                malicious_reasons.append(f"Token {addr} has hidden owner.")

            if _goplus_to_bool(result_for_addr["is_airdrop_scam"]):
                malicious_reasons.append(f"Token {addr} is an airdrop scam.")

            # Even sDai is flagged with this.
            # if _goplus_to_bool(result_for_addr["is_anti_whale"]):
            #     malicious_reasons.append(f"Token {addr} is anti-whale.")

            if _goplus_to_bool(result_for_addr["is_blacklisted"]):
                malicious_reasons.append(f"Token {addr} is blacklisted.")

            if _goplus_to_bool(result_for_addr["is_honeypot"]):
                malicious_reasons.append(f"Token {addr} is a honeypot.")

            if not _goplus_to_bool(result_for_addr["is_open_source"]):
                malicious_reasons.append(f"Token {addr} contract is not open source.")

            # Direct check for `False`, because None is fine here.
            if _goplus_to_bool(result_for_addr["is_true_token"]) is False:
                malicious_reasons.append(f"Token {addr} is not a true token.")

            if _goplus_to_bool(result_for_addr["is_whitelisted"]):
                malicious_reasons.append(f"Token {addr} is whitelisted.")

            if other_potential_risks := result_for_addr["other_potential_risks"]:
                malicious_reasons.append(
                    f"Token {addr} has other potential risks ({other_potential_risks})."
                )

            if _goplus_to_bool(result_for_addr["selfdestruct"]):
                malicious_reasons.append(f"Token {addr} has self-destruct function.")

        return _build_validation_result(self.name, self.description, malicious_reasons)


class GoPlusAddressSecurity(AbstractGuard):
    name = "Address security"
    description = "This guard checks the security of addresses involved in the transaction using GoPlus API."

    @observe(name="validate_safe_transaction_goplus_address_security")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
    ) -> ValidationResult | None:
        malicious_reasons: list[str] = []

        for addr in all_addresses_from_tx:
            result = goplus_address_security(addr)
            if result is None:
                continue

            logger.info(
                f"{self.name} response for address {addr}:\n\n{dict_to_point_list(result)}",
                streamlit=True,
            )

            if _goplus_to_bool(result["blacklist_doubt"]):
                malicious_reasons.append(f"Address {addr} is blacklisted.")

            if _goplus_to_bool(result["blackmail_activities"]):
                malicious_reasons.append(
                    f"Address {addr} has reported blackmail activities."
                )

            if _goplus_to_bool(result["cybercrime"]):
                malicious_reasons.append(
                    f"Address {addr} has reported cybercrime activities."
                )

            if _goplus_to_bool(result["darkweb_transactions"]):
                malicious_reasons.append(f"Address {addr} has darkweb transactions.")

            if _goplus_to_bool(result["fake_kyc"]):
                malicious_reasons.append(f"Address {addr} has fake KYC.")

            if _goplus_to_bool(result["fake_standard_interface"]):
                malicious_reasons.append(
                    f"Address {addr} has fake standard interface (usually happens with scam assets)."
                )

            if _goplus_to_bool(result["fake_token"]):
                malicious_reasons.append(f"Address {addr} has fake token.")

            if _goplus_to_bool(result["financial_crime"]):
                malicious_reasons.append(
                    f"Address {addr} has financial crime activities."
                )

            if _goplus_to_bool(result["gas_abuse"]):
                malicious_reasons.append(f"Address {addr} has gas abuse activities.")

            # Too many of false positives.
            # if _goplus_to_bool(result["honeypot_related_address"]):
            #     malicious_reasons.append(f"Address {addr} is related to a honeypot.")

            if _goplus_to_bool(result["malicious_mining_activities"]):
                malicious_reasons.append(
                    f"Address {addr} has malicious mining activities."
                )

            if _goplus_to_bool(result["mixer"]):
                malicious_reasons.append(f"Address {addr} is a mixer.")

            if _goplus_to_bool(result["money_laundering"]):
                malicious_reasons.append(
                    f"Address {addr} has money laundering activities."
                )

            # Too many of false positives.
            # if _goplus_to_bool(result["number_of_malicious_contracts_created"]):
            #     malicious_reasons.append(
            #         f"Address {addr} has created a number of malicious contracts."
            #     )

            if _goplus_to_bool(result["phishing_activities"]):
                malicious_reasons.append(
                    f"Address {addr} has reported phishing activities."
                )

            if _goplus_to_bool(result["reinit"]):
                malicious_reasons.append(
                    f"Address {addr} could be redeployed with a different code."
                )

            if _goplus_to_bool(result["sanctioned"]):
                malicious_reasons.append(f"Address {addr} is sanctioned.")

            if _goplus_to_bool(result["stealing_attack"]):
                malicious_reasons.append(
                    f"Address {addr} has stealing attack activities."
                )

        return _build_validation_result(self.name, self.description, malicious_reasons)


class GoPlusNftSecurity(AbstractGuard):
    name = "NFT security"
    description = "This guard checks the security of NFTs involved in the transaction using GoPlus API."

    @observe(name="validate_safe_transaction_goplus_nft_security")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
    ) -> ValidationResult | None:
        malicious_reasons: list[str] = []

        for addr in all_addresses_from_tx:
            result = goplus_nft_security(addr)
            if result is None:
                continue

            logger.info(
                f"{self.name} response for address {addr}:\n\n{dict_to_point_list(result)}",
                streamlit=True,
            )

            if _goplus_to_bool(result["malicious_nft_contract"]):
                malicious_reasons.append(
                    f"NFT contract {addr} malicious activity reported."
                )

            if not _goplus_to_bool(result["nft_open_source"]):
                malicious_reasons.append(f"NFT contract {addr} is not open source.")

            if (privileged_burn := result["privileged_burn"]) and _goplus_to_bool(
                privileged_burn["value"]
            ):
                malicious_reasons.append(
                    f"NFT contract {addr} has privileged burn function."
                )

            if (self_destruct := result["self_destruct"]) and _goplus_to_bool(
                self_destruct["value"]
            ):
                malicious_reasons.append(
                    f"NFT contract {addr} has self-destruct function."
                )

            if (
                transfer_without_approval := result["transfer_without_approval"]
            ) and _goplus_to_bool(transfer_without_approval["value"]):
                malicious_reasons.append(
                    f"NFT contract {addr} has transfer without approval function."
                )

        return _build_validation_result(self.name, self.description, malicious_reasons)


def _build_validation_result(
    name: str, description: str, malicious_reasons: list[str]
) -> ValidationResult:
    return ValidationResult(
        name=name,
        description=description,
        ok=not malicious_reasons,
        reason=" ".join(malicious_reasons) if malicious_reasons else "No issues found.",
    )


@db_cache(max_age=timedelta(days=3))
def goplus_token_security(
    address: ChecksumAddress, chain_id: int = RPCConfig().chain_id
) -> dict[str, Any] | None:
    # Used per-address instead of batching the call, so we can granualy cache the results and re-use across users.
    data = _goplus_call(
        lambda: Token().token_security(chain_id=f"{chain_id}", addresses=[address])
    )
    if data is None or not data.result:
        return None
    result: dict[str, Any] = data.result[address.lower()].to_dict()
    return result


@db_cache(max_age=timedelta(days=3))
def goplus_address_security(
    address: ChecksumAddress, chain_id: int = RPCConfig().chain_id
) -> dict[str, Any] | None:
    data = _goplus_call(
        lambda: Address().address_security(chain_id=f"{chain_id}", address=address)
    )
    if data is None or not data.result:
        return None
    result: dict[str, Any] = data.result.to_dict()
    return result


@db_cache(max_age=timedelta(days=3))
def goplus_nft_security(
    address: ChecksumAddress, chain_id: int = RPCConfig().chain_id
) -> dict[str, Any] | None:
    data = _goplus_call(
        lambda: Nft().nft_security(chain_id=f"{chain_id}", address=address)
    )
    if data is None or not data.result:
        return None
    result: dict[str, Any] = data.result.to_dict()
    return result


T = TypeVar("T")


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(max=10),
    retry=tenacity.retry_if_not_exception_type(GoPlusError),
)
def _goplus_call(f: Callable[[], T], retries: int = 3) -> T | None:
    response = f()
    code = int(response.code)  # type: ignore # GoPlus isn't typed.

    # See https://docs.gopluslabs.io/reference/api-status-code for possible status codes.
    if code == 1:
        return response

    elif code in (2, 4029) and retries > 0:
        logger.warning(f"Goplus rate limit hit, waiting a bit and trying again.")
        time.sleep(16)  # Sleep time based on the doc above.
        return _goplus_call(f, retries=retries - 1)

    elif code == 2 and retries <= 0:
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
