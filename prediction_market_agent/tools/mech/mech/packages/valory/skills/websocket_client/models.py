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

from aea.skills.base import Model

DEFAULT_WEBSOCKET_PROVIDER = "ws://localhost:8001"
DEFAULT_SUBSCRIPTION_ID = "websocket-subscription"


class Params(Model):
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""

        self.websocket_provider = kwargs.get(
            "websocket_provider", DEFAULT_WEBSOCKET_PROVIDER
        )
        self.subscription_id = kwargs.get("subscription_id", DEFAULT_SUBSCRIPTION_ID)
        super().__init__(*args, **kwargs)
