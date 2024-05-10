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

"""This package contains the rounds of SubscriptionUpdateAbciApp."""
from enum import Enum
from typing import Dict, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    DegenerateRound,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.subscription_abci.payloads import UpdateSubscriptionPayload


class Event(Enum):
    """SubscriptionUpdateAbciApp Events"""

    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
    NO_TX = "no_tx"
    ERROR = "error"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def most_voted_tx_hash(self) -> str:
        """Get the most_voted_tx_hash."""
        return cast(str, self.db.get_strict("most_voted_tx_hash"))


class UpdateSubscriptionRound(CollectSameUntilThresholdRound):
    """UpdateSubscriptionRound"""

    payload_class = UpdateSubscriptionPayload
    payload_attribute = "content"
    synchronized_data_class = SynchronizedData

    ERROR_PAYLOAD = "error"
    NO_TX_PAYLOAD = "no_tx"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload == self.ERROR_PAYLOAD:
                return self.synchronized_data, Event.ERROR

            if self.most_voted_payload == self.NO_TX_PAYLOAD:
                return self.synchronized_data, Event.NO_TX

            state = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(
                        SynchronizedData.most_voted_tx_hash
                    ): self.most_voted_payload,
                }
            )
            return state, Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY,

        return None


class FinishedWithTxRound(DegenerateRound):
    """FinishedWithTxRound"""

class FinishedWithoutTxRound(DegenerateRound):
    """FinishedWithoutTxRound"""


class SubscriptionUpdateAbciApp(AbciApp[Event]):
    """SubscriptionUpdateAbciApp

    Initial round: UpdateSubscriptionRound

    Initial states: {UpdateSubscriptionRound}

    Transition states:
        0. UpdateSubscriptionRound
            - done: 1.
            - no tx: 2.
            - error: 0.
            - no majority: 0.
        1. FinishedWithTxRound
        2. FinishedWithoutTxRound

    Final states: {FinishedWithTxRound, FinishedWithoutTxRound}

    Timeouts:
        round timeout: 60.0
    """

    initial_round_cls: AppState = UpdateSubscriptionRound
    initial_states: Set[AppState] = {UpdateSubscriptionRound}
    transition_function: AbciAppTransitionFunction = {
        UpdateSubscriptionRound: {
            Event.DONE: FinishedWithTxRound,
            Event.NO_TX: FinishedWithoutTxRound,
            Event.ERROR: UpdateSubscriptionRound,
            Event.NO_MAJORITY: UpdateSubscriptionRound,
        },
        FinishedWithTxRound: {},
        FinishedWithoutTxRound: {},
    }
    final_states: Set[AppState] = {
        FinishedWithTxRound,
        FinishedWithoutTxRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.ROUND_TIMEOUT: 60.0,
    }
    db_pre_conditions: Dict[AppState, Set[str]] = {
        UpdateSubscriptionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedWithTxRound: {"most_voted_tx_hash"},
        FinishedWithoutTxRound: set(),
    }
