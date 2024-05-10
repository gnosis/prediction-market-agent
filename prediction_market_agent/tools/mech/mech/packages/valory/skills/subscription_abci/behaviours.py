# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This package contains round behaviours of UpdateSubscriptionAbciApp."""
from abc import ABC
from typing import Any, Dict, Generator, List, Optional, Set, Type, cast

import openai  # noqa

from packages.valory.contracts.agent_mech.contract import (
    AgentMechContract,
)
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.subscription_abci.models import Params
from packages.valory.skills.subscription_abci.payloads import UpdateSubscriptionPayload
from packages.valory.skills.subscription_abci.rounds import (
    SynchronizedData,
    SubscriptionUpdateAbciApp,
    UpdateSubscriptionRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)

ZERO_ETHER_VALUE = 0
AUTO_GAS = SAFE_GAS = 0


class BaseSubscriptionBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the subscription_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)


class UpdateSubscriptionBehaviour(BaseSubscriptionBehaviour):
    """UpdateSubscriptionBehaviour"""

    matching_round: Type[AbstractRound] = UpdateSubscriptionRound

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            payload_content = yield from self.get_payload_content()
            sender = self.context.agent_address
            payload = UpdateSubscriptionPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def _should_update_subscription(
        self,
        mech_address: str,
        expected_subscription: str,
        expected_token_id: int,
    ) -> Generator[None, None, Optional[bool]]:
        """Check if the agent should update the subscription."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=mech_address,
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_subscription",
        )
        if (
                contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_subscription unsuccessful!: {contract_api_msg}"
            )
            return None

        actual_subscription_address = cast(str, contract_api_msg.state.body["nft"])
        actual_token_id = cast(int, contract_api_msg.state.body["token_id"])
        return actual_subscription_address != expected_subscription or actual_token_id != expected_token_id

    def _get_subscription_update_tx(
        self,
        mech_address: str,
        subscription_address: str,
        token_id: int,
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get the mech update hash tx."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=mech_address,
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_set_subscription_tx_data",
            subscription_address=subscription_address,
            token_id=token_id,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_set_subscription_tx_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": mech_address,
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }

    def get_subscription_update_txs(self) -> Generator[None, None, List[Dict[str, Any]]]:
        """Get the mech update hash tx."""
        txs = []
        for mech_address, subscription in self.params.mech_to_subscription.items():
            subscription_address, token_id = subscription
            should_update = yield from self._should_update_subscription(
                mech_address, subscription_address, token_id
            )
            if should_update is None:
                # something went wrong
                self.context.logger.warning(f"Could not check if subscription should be updated for {mech_address}.")
                continue
            if not should_update:
                # no need to update
                self.context.logger.info(f"No need to update subscription for {mech_address}.")
                continue

            tx = yield from self._get_subscription_update_tx(mech_address, subscription_address, token_id)
            if tx is None:
                # something went wrong
                self.context.logger.warning(f"Could not get subscription update tx for {mech_address}.")

            txs.append(tx)

        return txs

    def get_payload_content(self) -> Generator[None, None, str]:
        """Prepare the transaction"""
        txs = yield from self.get_subscription_update_txs()
        if len(txs) == 0:
            self.context.logger.info("No subscription update txs to send.")
            return UpdateSubscriptionRound.NO_TX_PAYLOAD

        multisend_tx_str = yield from self._to_multisend(txs)
        if multisend_tx_str is None:
            # something went wrong, respond with ERROR payload for now
            return UpdateSubscriptionRound.ERROR_PAYLOAD

        return multisend_tx_str

    def _to_multisend(
        self, transactions: List[Dict]
    ) -> Generator[None, None, Optional[str]]:
        """Transform payload to MultiSend."""
        multi_send_txs = []
        for transaction in transactions:
            transaction = {
                "operation": transaction.get("operation", MultiSendOperation.CALL),
                "to": transaction["to"],
                "value": transaction["value"],
                "data": transaction.get("data", b""),
            }
            multi_send_txs.append(transaction)

        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=self.params.multisend_address,
            contract_id=str(MultiSendContract.contract_id),
            contract_callable="get_tx_data",
            multi_send_txs=multi_send_txs,
        )
        if response.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Couldn't compile the multisend tx. "
                f"Expected performative {ContractApiMessage.Performative.RAW_TRANSACTION.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response
        multisend_data_str = cast(str, response.raw_transaction.body["data"])[2:]
        tx_data = bytes.fromhex(multisend_data_str)
        tx_hash = yield from self._get_safe_tx_hash(tx_data)
        if tx_hash is None:
            # something went wrong
            return None

        payload_data = hash_payload_to_hex(
            safe_tx_hash=tx_hash,
            ether_value=ZERO_ETHER_VALUE,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=self.params.multisend_address,
            data=tx_data,
            gas_limit=self.params.manual_gas_limit,
        )
        return payload_data

    def _get_safe_tx_hash(self, data: bytes) -> Generator[None, None, Optional[str]]:
        """
        Prepares and returns the safe tx hash.

        This hash will be signed later by the agents, and submitted to the safe contract.
        Note that this is the transaction that the safe will execute, with the provided data.

        :param data: the safe tx data.
        :return: the tx hash
        """
        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=self.params.multisend_address,  # we send the tx to the multisend address
            value=ZERO_ETHER_VALUE,
            data=data,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
        )

        if response.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Couldn't get safe hash. "
                f"Expected response performative {ContractApiMessage.Performative.STATE.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response hash
        tx_hash = cast(str, response.state.body["tx_hash"])[2:]
        return tx_hash


class UpdateSubscriptionRoundBehaviour(AbstractRoundBehaviour):
    """UpdateSubscriptionRoundBehaviour"""

    initial_behaviour_cls = UpdateSubscriptionBehaviour
    abci_app_cls = SubscriptionUpdateAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        UpdateSubscriptionBehaviour,
    }
