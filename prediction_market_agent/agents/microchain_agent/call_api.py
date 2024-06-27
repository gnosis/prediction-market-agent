import json
from typing import Optional

import requests
from microchain import Function


class CallAPI(Function):
    @property
    def description(self) -> str:
        """
        A property method that returns a description of the function, which can be used to make arbitrary API calls to any endpoint with any parameters. It supports different HTTP methods (GET, POST, etc.) and can include headers, query parameters, and request bodies.
        """
        return (
            "Use this function to make arbitrary API calls to any endpoint with any parameters. "
            "It supports different HTTP methods (GET, POST, etc.) and can include headers, "
            "query parameters, and request bodies. "
            "Params, data and headers needs to be passed as JSON strings."
        )

    @property
    def example_args(self) -> list[str]:
        """
        Returns a list of strings representing the example arguments for the API call.
        """
        return ["method", "url", "params", "data", "headers"]

    def __call__(
        self,
        method: str,
        url: str,
        params: Optional[str] = None,
        data: Optional[str] = None,
        headers: Optional[str] = None,
    ) -> str:
        """
        Sends an HTTP request using the specified method to the given URL with optional parameters, data, and headers.

        Args:
            method (str): The HTTP method to use for the request.
            url (str): The URL to send the request to.
            params (str, optional): Query parameters to include in the request. Defaults to None. If provided, needs to be in JSON string format.
            data (str, optional): Request body to include in the request. Defaults to None. If provided, needs to be in JSON string format.
            headers (str, optional): Headers to include in the request. Defaults to None. If provided, needs to be in JSON string format.

        Returns:
            str: The response string if the request was successful.
        """
        response = requests.request(
            method,
            url,
            params=json.loads(params) if params else None,
            data=json.loads(data) if data else None,
            headers=json.loads(headers) if headers else None,
        )
        response.raise_for_status()
        return response.text


API_FUNCTIONS: list[type[Function]] = [CallAPI]
