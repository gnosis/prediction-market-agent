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
"""Benchmarking for tools."""

import logging
from typing import Any, Callable, Dict, Union


PRICE_NUM_TOKENS = 1000


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }

    def __init__(self) -> None:
        """Initialize the callback."""
        self.cost_dict: Dict[str, Union[int, float]] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
        }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        return (
            tokens
            / PRICE_NUM_TOKENS
            * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]
        )

    def calculate_cost(
        self, tokens_type: str, model: str, token_counter: Callable, **kwargs: Any
    ) -> None:
        """Calculate the cost of a generation."""
        # Check if it its prompt or tokens are passed in
        prompt_key = f"{tokens_type}_prompt"
        token_key = f"{tokens_type}_tokens"
        if prompt_key in kwargs:
            tokens = token_counter(kwargs[prompt_key], model)
        elif token_key in kwargs:
            tokens = kwargs[token_key]
        else:
            logging.warning(f"No {token_key}_tokens or {tokens_type}_prompt found.")
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, token_counter: Callable, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            self.calculate_cost("input", model, token_counter, **kwargs)
            self.calculate_cost("output", model, token_counter, **kwargs)
            self.cost_dict["total_tokens"] = (
                self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            )
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"] + self.cost_dict["output_cost"]
            )
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")
