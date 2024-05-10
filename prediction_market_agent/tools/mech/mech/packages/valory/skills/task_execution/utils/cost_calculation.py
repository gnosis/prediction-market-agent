# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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
"""Calculate the cost for tools."""
import logging
import math
from typing import Any, Dict, cast

from packages.valory.skills.task_execution import PUBLIC_ID


_logger = logging.getLogger(
    f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.utils.cost_calculation"
)

DEFAULT_PRICE = 1


def get_cost_for_done_task(
    done_task: Dict[str, Any], fallback_price: int = DEFAULT_PRICE
) -> int:
    """Get the cost for a done task."""
    cost_dict = done_task.get("cost_dict", {})
    if cost_dict == {}:
        _logger.warning(f"Cost dict not found in done task {done_task['request_id']}.")
        return fallback_price
    total_cost = cost_dict.get("total_cost", None)
    if total_cost is None:
        _logger.warning(
            f"Total cost not found in cost dict {cost_dict} for {done_task['request_id']}."
        )
        return fallback_price

    total_cost = cast(float, total_cost)
    # 0.01 token (ex. xDAI/USDC) -> 1 NFT credit
    cost_in_credits = math.ceil(total_cost * 100)
    return cost_in_credits
