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

"""This package contains a custom Loader for the ipfs connection."""

from typing import Any


class AnyToolAsTask:
    """AnyToolAsTask"""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task."""
        tool_py = kwargs.pop("tool_py")
        callable_method = kwargs.pop("callable_method")
        local_namespace: Any = {}
        exec(tool_py, local_namespace)  # pylint: disable=W0122  # nosec
        method = local_namespace[callable_method]
        return method(*args, **kwargs)
