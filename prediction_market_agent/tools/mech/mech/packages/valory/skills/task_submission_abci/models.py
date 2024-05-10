# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This module contains the shared state for the abci skill of TaskExecutionAbciApp."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from aea.exceptions import enforce

from packages.valory.skills.abstract_round_abci.base import AbciApp
from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.abstract_round_abci.models import TypeCheckMixin
from packages.valory.skills.abstract_round_abci.utils import check_type
from packages.valory.skills.task_submission_abci.rounds import TaskSubmissionAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls: Type[AbciApp] = TaskSubmissionAbciApp


@dataclass
class MutableParams(TypeCheckMixin):
    """Collection for the mutable parameters."""

    latest_metadata_hash: Optional[bytes] = None


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.task_wait_timeout = self._ensure("task_wait_timeout", kwargs, float)
        self.service_endpoint_base = self._ensure("service_endpoint_base", kwargs, str)
        self.multisend_address = self._ensure_get("multisend_address", kwargs, str)
        self.agent_registry_address = self._ensure(
            "agent_registry_address", kwargs, str
        )
        self.agent_id: int = self._ensure("agent_id", kwargs, int)
        self.metadata_hash: str = self._ensure("metadata_hash", kwargs, str)
        self.task_mutable_params = MutableParams()
        self.manual_gas_limit = self._ensure_get("manual_gas_limit", kwargs, int)
        self.service_owner_share = self._ensure("service_owner_share", kwargs, float)
        self.profit_split_freq = self._ensure("profit_split_freq", kwargs, int)
        self.agent_mech_contract_addresses = self._ensure(
            "agent_mech_contract_addresses", kwargs, list
        )
        self.hash_checkpoint_address = self._ensure(
            "hash_checkpoint_address", kwargs, str
        )
        self.minimum_agent_balance = self._ensure("minimum_agent_balance", kwargs, int)
        self.agent_funding_amount = self._ensure("agent_funding_amount", kwargs, int)
        super().__init__(*args, **kwargs)

    @classmethod
    def _ensure_get(cls, key: str, kwargs: Dict, type_: Any) -> Any:
        """Ensure that the parameters are set, and return them without popping the key."""
        enforce("skill_context" in kwargs, "Only use on models!")
        skill_id = kwargs["skill_context"].skill_id
        enforce(
            key in kwargs,
            f"'{key!r}' of type '{type_!r}' required, but it is not set in `models.params.args` of `skill.yaml` of `{skill_id}`",
        )
        value = kwargs.get(key, None)
        try:
            check_type(key, value, type_)
        except TypeError:  # pragma: nocover
            enforce(
                False,
                f"'{key!r}' must be a {type_!r}, but type {type(value)} was found in `models.params.args` of `skill.yaml` of `{skill_id}`",
            )
        return value


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
