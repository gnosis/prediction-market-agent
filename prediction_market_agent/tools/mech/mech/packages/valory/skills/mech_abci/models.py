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

"""This module contains the shared state for the abci skill of Mech."""
from typing import Any

from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.mech_abci.composition import MechAbciApp
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.subscription_abci.models import Params as SubscriptionParams
from packages.valory.skills.task_submission_abci.models import (
    Params as TaskExecutionAbciParams,
)
from packages.valory.skills.task_submission_abci.models import (
    SharedState as TaskExecSharedState,
)
from packages.valory.skills.task_submission_abci.rounds import (
    Event as TaskExecutionEvent,
)
from packages.valory.skills.termination_abci.models import TerminationParams
from packages.valory.skills.transaction_settlement_abci.rounds import (
    Event as TransactionSettlementEvent,
)


TaskExecutionParams = TaskExecutionAbciParams


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class RandomnessApi(ApiSpecs):
    """A model that wraps ApiSpecs for randomness api specifications."""


MARGIN = 5


class SharedState(TaskExecSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = MechAbciApp

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the shared state."""
        self.last_processed_request_block_number: int = 0
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Set up."""
        super().setup()

        MechAbciApp.event_to_timeout[
            TaskExecutionEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            TaskExecutionEvent.TASK_EXECUTION_ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            ResetPauseEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            TransactionSettlementEvent.ROUND_TIMEOUT
        ] = self.context.params.round_timeout_seconds

        MechAbciApp.event_to_timeout[
            TransactionSettlementEvent.VALIDATE_TIMEOUT
        ] = self.context.params.validate_timeout

        MechAbciApp.event_to_timeout[
            TransactionSettlementEvent.FINALIZE_TIMEOUT
        ] = self.context.params.finalize_timeout

        MechAbciApp.event_to_timeout[ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT] = (
            self.context.params.reset_pause_duration + MARGIN
        )


class Params(TaskExecutionParams, SubscriptionParams, TerminationParams):  # type: ignore
    """A model to represent params for multiple abci apps."""
