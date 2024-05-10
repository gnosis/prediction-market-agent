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
"""This module contains helpers for IPFS interaction."""
from typing import Any, Dict, Tuple

import yaml
from aea.helpers.cid import CID
from multibase import multibase
from multicodec import multicodec


CID_PREFIX = "f01701220"


def get_ipfs_file_hash(data: bytes) -> str:
    """Get hash from bytes"""
    try:
        return str(CID.from_string(data.decode()))
    except Exception:  # noqa
        # if something goes wrong, fallback to sha256
        file_hash = data.hex()
        file_hash = CID_PREFIX + file_hash
        file_hash = str(CID.from_string(file_hash))
        return file_hash


def to_multihash(hash_string: str) -> str:
    """To multihash string."""
    # Decode the Base32 CID to bytes
    cid_bytes = multibase.decode(hash_string)
    # Remove the multicodec prefix (0x01) from the bytes
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    # Convert the multihash bytes to a hexadecimal string
    hex_multihash = multihash_bytes.hex()
    return hex_multihash[6:]


class ComponentPackageLoader:
    """Component package loader."""

    @staticmethod
    def load(serialized_objects: Dict[str, str]) -> Tuple[Dict[str, Any], str, str]:
        """
        Load a custom component package.

        :param serialized_objects: the serialized objects.
        :return: the component.yaml, entry_point.py and callable as tuple.
        """
        # the package MUST contain a component.yaml file
        if "component.yaml" not in serialized_objects:
            raise ValueError(
                "Invalid component package. "
                "The package MUST contain a component.yaml."
            )

        # load the component.yaml file
        component_yaml = yaml.safe_load(serialized_objects["component.yaml"])
        if "entry_point" not in component_yaml or "callable" not in component_yaml:
            raise ValueError(
                "Invalid component package. "
                "The component.yaml file MUST contain the 'entry_point' and 'callable' keys."
            )

        # the name of the script that needs to be executed
        entry_point_name = component_yaml["entry_point"]

        # load the script
        if entry_point_name not in serialized_objects:
            raise ValueError(
                f"Invalid component package. "
                f"{entry_point_name} is not present in the component package."
            )
        entry_point = serialized_objects[entry_point_name]

        # the method that needs to be called
        callable_method = component_yaml["callable"]

        return component_yaml, entry_point, callable_method
