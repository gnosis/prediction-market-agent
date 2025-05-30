import json
import os
from pprint import pprint
from typing import Any, Dict, List

import typer

from prediction_market_agent.agents.safe_watch_agent.safe_utils import (
    find_addresses_in_nested_structure,
)


def main(directory_path: str) -> None:
    """
    Helper script used to extract all addreses mentioned in the https://github.com/safe-global/safe-deployments/tree/main/src/assets.
    """
    all_addresses: set[str] = set()
    json_data = load_json_files_from_directory(directory_path)
    for data in json_data:
        addresses = find_addresses_in_nested_structure(data)
        all_addresses.update(addresses)
    pprint(all_addresses)


def load_json_files_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    json_files_data = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as json_file:
                    try:
                        data = json.load(json_file)
                        json_files_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {filename}: {e}")
    return json_files_data


# Example usage
if __name__ == "__main__":
    typer.run(main)
