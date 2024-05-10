# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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

"""This module contains the class to connect to the Service Registry contract."""

import logging
from typing import Any, Optional

from aea.common import JSONLike
from aea.configurations.base import PublicId
from aea.contracts.base import Contract
from aea.crypto.base import LedgerApi


PUBLIC_ID = PublicId.from_str("valory/agent_registry:0.1.0")

AGENT_UNIT_TYPE = 1
UNIT_HASH_PREFIX = "0x{metadata_hash}"

_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract"
)


class AgentRegistryContract(Contract):
    """The Agent Registry contract."""

    contract_id = PUBLIC_ID

    @classmethod
    def get_raw_transaction(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[JSONLike]:
        """Get the Safe transaction."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_raw_message(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[bytes]:
        """Get raw message."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_state(
        cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any
    ) -> Optional[JSONLike]:
        """Get state."""
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_create_events(  # pragma: nocover
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        receipt: JSONLike,
    ) -> Optional[int]:
        """Returns `CreateUnit` event filter."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        return contract_interface.events.CreateUnit().process_receipt(receipt)

    @classmethod
    def get_update_hash_events(  # pragma: nocover
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        receipt: JSONLike,
    ) -> Optional[int]:
        """Returns `CreateUnit` event filter."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        return contract_interface.events.UpdateUnitHash().process_receipt(receipt)

    @classmethod
    def get_token_uri(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        token_id: int,
    ) -> str:
        """Returns the latest metadata URI for a component."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        _, hash_updates = contract_interface.functions.getHashes(token_id).call()
        if len(hash_updates) > 0:  # pragma: nocover
            *_, latest_hash = hash_updates
            uri = f"https://gateway.autonolas.tech/ipfs/f01701220{latest_hash.hex()}"
        else:
            uri = contract_interface.functions.tokenURI(token_id).call()
        return uri

    @classmethod
    def get_token_hash(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        token_id: int,
    ) -> JSONLike:
        """Returns the latest metadata URI for a component."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        _, hash_updates = contract_interface.functions.getHashes(token_id).call()
        if len(hash_updates) > 0:  # pragma: nocover
            *_, latest_hash = hash_updates
            return dict(data=latest_hash.hex())
        _logger.warning(f"No metadata hash updates found for {token_id} on {contract_address}.")
        return dict(data=None)

    @classmethod
    def get_update_hash_tx_data(
        cls,
        ledger_api: LedgerApi,
        contract_address: str,
        token_id: int,
        metadata_hash: bytes,
    ) -> JSONLike:
        """Returns the transaction to update the metadata hash."""
        contract_interface = cls.get_instance(
            ledger_api=ledger_api,
            contract_address=contract_address,
        )
        data = contract_interface.encodeABI(
            fn_name="updateHash", args=[token_id, metadata_hash]
        )
        return dict(data=data)
