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

"""This package contains the implementation of ."""
import json
import threading
import time
from asyncio import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from aea.helpers.cid import to_v1
from aea.mail.base import EnvelopeContext
from aea.protocols.base import Message
from aea.protocols.dialogue.base import Dialogue
from aea.skills.behaviours import SimpleBehaviour
from eth_abi import encode

from packages.valory.connections.ipfs.connection import IpfsDialogues
from packages.valory.connections.ipfs.connection import PUBLIC_ID as IPFS_CONNECTION_ID
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.valory.connections.p2p_libp2p_client.connection import (
    PUBLIC_ID as P2P_CLIENT_PUBLIC_ID,
)
from packages.valory.contracts.agent_mech.contract import AgentMechContract
from packages.valory.protocols.acn_data_share import AcnDataShareMessage
from packages.valory.protocols.acn_data_share.dialogues import AcnDataShareDialogues
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ipfs import IpfsMessage
from packages.valory.protocols.ipfs.dialogues import IpfsDialogue
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.task_execution.models import Params
from packages.valory.skills.task_execution.utils.benchmarks import TokenCounterCallback
from packages.valory.skills.task_execution.utils.cost_calculation import (
    get_cost_for_done_task,
)
from packages.valory.skills.task_execution.utils.ipfs import (
    ComponentPackageLoader,
    get_ipfs_file_hash,
    to_multihash,
)
from packages.valory.skills.task_execution.utils.task import AnyToolAsTask


PENDING_TASKS = "pending_tasks"
DONE_TASKS = "ready_tasks"
DONE_TASKS_LOCK = "lock"
GNOSIS_CHAIN = "gnosis"

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


class TaskExecutionBehaviour(SimpleBehaviour):
    """A class to execute tasks."""

    def __init__(self, **kwargs: Any):
        """Initialise the agent."""
        super().__init__(**kwargs)
        # we only want to execute one task at a time, for the time being
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._executing_task: Optional[Dict[str, Any]] = None
        self._tools_to_file_hash: Dict[str, str] = {}
        self._all_tools: Dict[str, Tuple[str, str]] = {}
        self._inflight_tool_req: Optional[str] = None
        self._done_task: Optional[Dict[str, Any]] = None
        self._last_polling: Optional[float] = None
        self._invalid_request = False
        self._async_result: Optional[Future] = None

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up TaskExecutionBehaviour")
        self._tools_to_file_hash = {
            value: key
            for key, values in self.params.file_hash_to_tools.items()
            for value in values
        }

    def act(self) -> None:
        """Implement the act."""
        self._download_tools()
        self._execute_task()
        self._check_for_new_reqs()

    @property
    def done_tasks_lock(self) -> threading.Lock:
        """Get done_tasks_lock."""
        return self.context.shared_state[DONE_TASKS_LOCK]

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def request_id_to_num_timeouts(self) -> Dict[int, int]:
        """Maps the request id to the number of times it has timed out."""
        return self.params.request_id_to_num_timeouts

    def count_timeout(self, request_id: int) -> None:
        """Increase the timeout for a request."""
        self.request_id_to_num_timeouts[request_id] += 1

    def timeout_limit_reached(self, request_id: int) -> bool:
        """Check if the timeout limit has been reached."""
        return self.params.timeout_limit <= self.request_id_to_num_timeouts[request_id]

    @property
    def pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending_tasks."""
        return self.context.shared_state[PENDING_TASKS]

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """Get done_tasks."""
        return self.context.shared_state[DONE_TASKS]

    def _should_poll(self) -> bool:
        """If we should poll the contract."""
        if self._last_polling is None:
            return True
        return self._last_polling + self.params.polling_interval <= time.time()

    def _is_executing_task_ready(self) -> bool:
        """Check if the executing task is ready."""
        if self._executing_task is None or self._async_result is None:
            return False
        return self._async_result.done()

    def _has_executing_task_timed_out(self) -> bool:
        """Check if the executing task timed out."""
        if self._executing_task is None:
            return False
        timeout_deadline = self._executing_task.get("timeout_deadline", None)
        if timeout_deadline is None:
            return False
        return timeout_deadline <= time.time()

    def _get_executing_task_result(self) -> Any:
        """Get the executing task result."""
        if self._executing_task is None:
            raise ValueError("Executing task is None")
        if self._invalid_request:
            return None
        try:
            async_result = cast(Future, self._async_result)
            return async_result.result()
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(
                "Exception raised while executing task: {}".format(str(e))
            )
            return None

    def _download_tools(self) -> None:
        """Download tools."""
        if self._inflight_tool_req is not None:
            # there already is a req in flight
            return
        if len(self._tools_to_file_hash) == len(self._all_tools):
            # we already have all the tools
            return
        for tool, file_hash in self._tools_to_file_hash.items():
            if tool in self._all_tools:
                continue
            # read one at a time
            ipfs_msg, message = self._build_ipfs_get_file_req(file_hash)
            self._inflight_tool_req = tool
            self.send_message(ipfs_msg, message, self._handle_get_tool)
            return

    def _handle_get_tool(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle get tool response"""
        _component_yaml, tool_py, callable_method = ComponentPackageLoader.load(
            message.files
        )
        tool_req = cast(str, self._inflight_tool_req)
        self._all_tools[tool_req] = tool_py, callable_method
        self._inflight_tool_req = None

    def _populate_from_block(self) -> None:
        """Populate from_block"""
        ledger_api_msg, _ = self.context.ledger_dialogues.create(
            performative=LedgerApiMessage.Performative.GET_STATE,
            callable="get_block",
            kwargs=LedgerApiMessage.Kwargs(dict(block_identifier="latest")),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
            args=(),
        )
        self.context.outbox.put_message(message=ledger_api_msg)
        self.params.in_flight_req = True

    def _check_for_new_reqs(self) -> None:
        """Check for new reqs."""
        if self.params.in_flight_req or not self._should_poll():
            # do nothing if there is an in flight request
            # or if we should not poll yet
            return

        if self.params.from_block is None:
            # set the initial from block
            self._populate_from_block()
            return
        contract_api_msg, _ = self.context.contract_dialogues.create(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.agent_mech_contract_addresses[0],
            contract_id=str(AgentMechContract.contract_id),
            callable="get_multiple_undelivered_reqs",
            kwargs=ContractApiMessage.Kwargs(
                dict(
                    from_block=self.params.from_block,
                    chain_id=GNOSIS_CHAIN,
                    contract_addresses=self.params.agent_mech_contract_addresses,
                    max_block_window=self.params.max_block_window,
                )
            ),
            counterparty=LEDGER_API_ADDRESS,
            ledger_id=self.context.default_ledger_id,
        )
        self.context.outbox.put_message(message=contract_api_msg)
        self.params.in_flight_req = True
        self._last_polling = time.time()

    def _execute_task(self) -> None:
        """Execute tasks."""
        # check if there is a task already executing
        if self.params.in_flight_req:
            # there is an in flight request
            return

        if self._executing_task is not None:
            if self._is_executing_task_ready() or self._invalid_request:
                task_result = self._get_executing_task_result()
                self._handle_done_task(task_result)
            elif self._has_executing_task_timed_out():
                self._handle_timeout_task()
            return

        if len(self.pending_tasks) == 0:
            # not tasks (requests) to execute
            return

        # create new task
        task_data = self.pending_tasks.pop(0)
        self.context.logger.info(f"Preparing task with data: {task_data}")
        self._executing_task = task_data
        task_data_ = task_data["data"]
        ipfs_hash = get_ipfs_file_hash(task_data_)
        self.context.logger.info(f"IPFS hash: {ipfs_hash}")
        ipfs_msg, message = self._build_ipfs_get_file_req(ipfs_hash)
        self.send_message(ipfs_msg, message, self._handle_get_task)

    def send_message(
        self, msg: Message, dialogue: Dialogue, callback: Callable
    ) -> None:
        """Send message."""
        self.context.outbox.put_message(message=msg)
        nonce = dialogue.dialogue_label.dialogue_reference[0]
        self.params.req_to_callback[nonce] = callback
        self.params.in_flight_req = True

    def _handle_done_task(self, task_result: Any) -> None:
        """Handle done tasks"""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", None)
        request_id_nonce = executing_task.get("requestIdWithNonce", None)
        mech_address = executing_task.get("contract_address", None)
        tool = executing_task.get("tool", None)
        model = executing_task.get("model", None)
        response = {"requestId": req_id, "result": "Invalid response"}
        task_executor = self.context.agent_address
        self._done_task = {
            "request_id": req_id,
            "mech_address": mech_address,
            "task_executor_address": task_executor,
            "tool": tool,
            "request_id_nonce": request_id_nonce,
        }
        if task_result is not None:
            # task succeeded
            deliver_msg, prompt, transaction, counter_callback = task_result
            cost_dict = {}
            if counter_callback is not None:
                cost_dict = cast(TokenCounterCallback, counter_callback).cost_dict
            metadata = {
                "model": model,
                "tool": tool,
            }
            response = {
                **response,
                "result": deliver_msg,
                "prompt": prompt,
                "cost_dict": cost_dict,
                "metadata": metadata,
            }
            self._done_task["transaction"] = transaction

        self.context.logger.info(f"Task result for request {req_id}: {task_result}")
        msg, dialogue = self._build_ipfs_store_file_req(
            {str(req_id): json.dumps(response)}
        )
        self.send_message(msg, dialogue, self._handle_store_response)

    def _handle_timeout_task(self) -> None:
        """Handle timeout tasks"""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id = executing_task.get("requestId", None)
        self.count_timeout(req_id)
        self.context.logger.info(f"Task timed out for request {req_id}")
        self.context.logger.info(
            f"Task {req_id} has timed out {self.request_id_to_num_timeouts[req_id]} times"
        )
        async_result = cast(Future, self._async_result)
        async_result.cancel()
        if not self.timeout_limit_reached(req_id):
            # added to end of queue
            self.context.logger.info(f"Adding task {req_id} to the end of the queue")
            self.pending_tasks.append(executing_task)
            self._executing_task = None
            return None

        self.context.logger.info(
            f"Task {req_id} has reached the timeout limit of{self.params.timeout_limit}. "
            f"It won't be added to the end of the queue again."
        )
        task_result = (
            f"Task timed out {self.params.timeout_limit} times during execution. ",
            "",
            None,
            None,
        )
        self._handle_done_task(task_result)

    def _handle_get_task(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a task request."""
        task_data = [json.loads(content) for content in message.files.values()][0]
        is_data_valid = (
            task_data
            and isinstance(task_data, dict)
            and "prompt" in task_data
            and "tool" in task_data
        )  # pylint: disable=C0301
        if is_data_valid and task_data["tool"] in self._tools_to_file_hash:
            self._prepare_task(task_data)
        elif is_data_valid:
            tool = task_data["tool"]
            executing_task = cast(Dict[str, Any], self._executing_task)
            executing_task["tool"] = tool
            self.context.logger.warning(f"Tool {tool} is not valid.")
            self._invalid_request = True
        else:
            self.context.logger.warning("Data for task is not valid.")
            self._invalid_request = True

    def _submit_task(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        """Submit a task."""
        try:
            return self._executor.submit(fn, *args, **kwargs)  # type: ignore
        except BrokenProcessPool:
            self.context.logger.warning("Executor is broken. Restarting...")
            # stop the current executor
            self._executor.shutdown(wait=False)
            # create a new executor
            self._executor = ProcessPoolExecutor(max_workers=1)
            # try to run the task again
            return self._executor.submit(fn, *args, **kwargs)  # type: ignore

    def _prepare_task(self, task_data: Dict[str, Any]) -> None:
        """Prepare the task."""
        tool_task = AnyToolAsTask()
        tool_py, callable_method = self._all_tools[task_data["tool"]]
        task_data["tool_py"] = tool_py
        task_data["callable_method"] = callable_method
        task_data["api_keys"] = self.params.api_keys
        task_data["counter_callback"] = TokenCounterCallback()
        future = self._submit_task(tool_task.execute, **task_data)
        executing_task = cast(Dict[str, Any], self._executing_task)
        executing_task["timeout_deadline"] = time.time() + self.params.task_deadline
        executing_task["tool"] = task_data["tool"]
        executing_task["model"] = task_data.get("model", None)
        self._async_result = cast(Optional[Future], future)

    def _build_ipfs_message(
        self,
        performative: IpfsMessage.Performative,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds an IPFS message."""
        ipfs_dialogues = cast(IpfsDialogues, self.context.ipfs_dialogues)
        message, dialogue = ipfs_dialogues.create(
            counterparty=str(IPFS_CONNECTION_ID),
            performative=performative,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_store_file_req(  # pylint: disable=too-many-arguments
        self,
        filename_to_obj: Dict[str, str],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """Builds a STORE_FILES ipfs message."""
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.STORE_FILES,  # type: ignore
            files=filename_to_obj,
            timeout=timeout,
            **kwargs,
        )
        return message, dialogue

    def _build_ipfs_get_file_req(
        self,
        ipfs_hash: str,
        timeout: Optional[float] = None,
    ) -> Tuple[IpfsMessage, IpfsDialogue]:
        """
        Builds a GET_FILES IPFS request.

        :param ipfs_hash: the ipfs hash of the file/dir to download.
        :param timeout: timeout for the request.
        :returns: the ipfs message, and its corresponding dialogue.
        """
        message, dialogue = self._build_ipfs_message(
            performative=IpfsMessage.Performative.GET_FILES,  # type: ignore
            ipfs_hash=ipfs_hash,
            timeout=timeout,
        )
        return message, dialogue

    def _handle_store_response(self, message: IpfsMessage, dialogue: Dialogue) -> None:
        """Handle the response from ipfs for a store response request."""
        executing_task = cast(Dict[str, Any], self._executing_task)
        req_id, sender = (
            executing_task["requestId"],
            executing_task["sender"],
        )
        ipfs_hash = to_v1(message.ipfs_hash)
        self.context.logger.info(
            f"Response for request {req_id} stored on IPFS with hash {ipfs_hash}."
        )
        self.send_data_via_acn(
            sender_address=sender,
            request_id=str(req_id),
            data=ipfs_hash,
        )
        done_task = cast(Dict[str, Any], self._done_task)
        task_result = to_multihash(ipfs_hash)
        cost = get_cost_for_done_task(done_task)
        self.context.logger.info(f"Cost for task {req_id}: {cost}")
        mech_config = self.params.mech_to_config[done_task["mech_address"]]
        if mech_config.use_dynamic_pricing:
            self.context.logger.info(f"Dynamic pricing is enabled for task {req_id}.")
            task_result = encode(
                ["uint256", "bytes"], [cost, bytes.fromhex(task_result)]
            ).hex()

        done_task["task_result"] = task_result
        # add to done tasks, in thread safe way
        with self.done_tasks_lock:
            self.done_tasks.append(done_task)
        # reset tasks
        self._executing_task = None
        self._done_task = None
        self._invalid_request = False

    def send_data_via_acn(
        self,
        sender_address: str,
        request_id: str,
        data: Any,
    ) -> None:
        """Handle callbacks."""
        self.context.logger.info(
            f"Sending data to {sender_address} via ACN for request ID {request_id}"
        )
        response, _ = cast(
            AcnDataShareDialogues, self.context.acn_data_share_dialogues
        ).create(
            counterparty=sender_address,
            performative=AcnDataShareMessage.Performative.DATA,
            request_id=request_id,
            content=data,
        )
        self.context.outbox.put_message(
            message=response,
            context=EnvelopeContext(connection_id=P2P_CLIENT_PUBLIC_ID),
        )
