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

"""This package contains the rounds of TaskSubmissionAbciApp."""
import json
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.task_submission_abci.payloads import (
    TaskPoolingPayload,
    TransactionPayload,
)


class Event(Enum):
    """TaskSubmissionAbciApp Events"""

    TASK_EXECUTION_ROUND_TIMEOUT = "task_execution_round_timeout"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
    NO_TASKS = "no_tasks"
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

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Done tasks."""
        return cast(List[Dict[str, Any]], self.db.get("done_tasks", []))


class TaskPoolingRound(CollectionRound):
    """TaskPoolingRound"""

    payload_class = TaskPoolingPayload
    synchronized_data_class = SynchronizedData

    move_forward_payload: Optional[TaskPoolingPayload] = None

    ERROR_PAYLOAD = "ERROR"

    @property
    def collection_threshold_reached(
        self,
    ) -> bool:
        """Check that the collection threshold has been reached."""
        return len(self.collection) >= self.synchronized_data.consensus_threshold

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.collection_threshold_reached:
            all_done_tasks = []
            for payload in self.collection.values():
                done_tasks_str = cast(TaskPoolingPayload, payload).content
                done_tasks = json.loads(done_tasks_str)
                all_done_tasks.extend(done_tasks)
            all_done_tasks = sorted(all_done_tasks, key=lambda x: x["request_id"])
            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=SynchronizedData,
                **{
                    get_name(SynchronizedData.done_tasks): all_done_tasks,
                }
            )
            if len(all_done_tasks) > 0:
                return synchronized_data, Event.DONE
            return synchronized_data, Event.NO_TASKS

        return None


class TransactionPreparationRound(CollectSameUntilThresholdRound):
    """TransactionPreparationRound"""

    payload_class = TransactionPayload
    payload_attribute = "content"
    synchronized_data_class = SynchronizedData

    ERROR_PAYLOAD = "error"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload == self.ERROR_PAYLOAD:
                return (
                    self.synchronized_data.update(
                        synchronized_data_class=SynchronizedData,
                        **{
                            get_name(SynchronizedData.done_tasks): [],
                        }
                    ),
                    Event.ERROR,
                )

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
            # in case we cant submit this tx, we need to make sure we don't account the tasks as done
            return (
                self.synchronized_data.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.done_tasks): [],
                    }
                ),
                Event.NO_MAJORITY,
            )

        return None


class FinishedTaskPoolingRound(DegenerateRound):
    """FinishedTaskPoolingRound"""


class FinishedTaskExecutionWithErrorRound(DegenerateRound):
    """FinishedTaskExecutionWithErrorRound"""


class FinishedWithoutTasksRound(DegenerateRound):
    """FinishedWithoutTasksRound"""


class TaskSubmissionAbciApp(AbciApp[Event]):
    """TaskSubmissionAbciApp

    Initial round: TaskPoolingRound

    Initial states: {TaskPoolingRound}

    Transition states:
        0. TaskPoolingRound
            - done: 1.
            - no tasks: 4.
            - round timeout: 0.
        1. TransactionPreparationRound
            - done: 2.
            - error: 3.
            - no majority: 3.
            - task execution round timeout: 1.
        2. FinishedTaskPoolingRound
        3. FinishedTaskExecutionWithErrorRound
        4. FinishedWithoutTasksRound

    Final states: {FinishedTaskExecutionWithErrorRound, FinishedTaskPoolingRound, FinishedWithoutTasksRound}

    Timeouts:
        task execution round timeout: 60.0
        round timeout: 60.0
    """

    initial_round_cls: AppState = TaskPoolingRound
    initial_states: Set[AppState] = {TaskPoolingRound}
    transition_function: AbciAppTransitionFunction = {
        TaskPoolingRound: {
            Event.DONE: TransactionPreparationRound,
            Event.NO_TASKS: FinishedWithoutTasksRound,
            Event.ROUND_TIMEOUT: TaskPoolingRound,
        },
        TransactionPreparationRound: {
            Event.DONE: FinishedTaskPoolingRound,
            Event.ERROR: FinishedTaskExecutionWithErrorRound,
            Event.NO_MAJORITY: FinishedTaskExecutionWithErrorRound,
            Event.TASK_EXECUTION_ROUND_TIMEOUT: TransactionPreparationRound,
        },
        FinishedTaskPoolingRound: {},
        FinishedTaskExecutionWithErrorRound: {},
        FinishedWithoutTasksRound: {},
    }
    final_states: Set[AppState] = {
        FinishedTaskPoolingRound,
        FinishedWithoutTasksRound,
        FinishedTaskExecutionWithErrorRound,
    }
    event_to_timeout: EventToTimeout = {
        Event.TASK_EXECUTION_ROUND_TIMEOUT: 60.0,
        Event.ROUND_TIMEOUT: 60.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset(
        [get_name(SynchronizedData.done_tasks)]
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        TaskPoolingRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedTaskPoolingRound: {"most_voted_tx_hash"},
        FinishedTaskExecutionWithErrorRound: set(),
        FinishedWithoutTasksRound: set(),
    }
