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

from packages.valory.skills.websocket_client.models import Params as BaseParams


DEFAULT_WEBSOCKET_PROVIDER = "ws://localhost:8001"
DEFAULT_CONTRACT_ADDRESS = "0xFf82123dFB52ab75C417195c5fDB87630145ae81"


class Params(BaseParams):
    """A model to represent params for multiple abci apps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        super().__init__(*args, **kwargs)

        self.use_polling = kwargs.get("use_polling", False)
        self.contract_address = kwargs.get("contract_address", DEFAULT_CONTRACT_ADDRESS)
