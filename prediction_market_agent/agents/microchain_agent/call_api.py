import requests

class CallAPI:
    def __init__(self) -> None:
        """
        Initializes the class instance.
        """
        super().__init__()

    @property
    def description(self) -> str:
        """
        A property method that returns a description of the function, which can be used to make arbitrary API calls to any endpoint with any parameters. It supports different HTTP methods (GET, POST, etc.) and can include headers, query parameters, and request bodies.
        """
        return (
            "Use this function to make arbitrary API calls to any endpoint with any parameters. "
            "It supports different HTTP methods (GET, POST, etc.) and can include headers, "
            "query parameters, and request bodies."
        )

    @property
    def example_args(self) -> list[str]:
        """
        Returns a list of strings representing the example arguments for the API call.

        """
        return ["method", "url", "params", "data", "headers"]

    def __call__(self, method: str, url: str, params: dict = None, data: dict = None, headers: dict = None) -> str:
        """
        Sends an HTTP request using the specified method to the given URL with optional parameters, data, and headers.

        Args:
            method (str): The HTTP method to use for the request.
            url (str): The URL to send the request to.
            params (dict, optional): Query parameters to include in the request. Defaults to None.
            data (dict, optional): Request body to include in the request. Defaults to None.
            headers (dict, optional): Headers to include in the request. Defaults to None.

        Returns:
            str: The response string if the request was successful, or the error message if an exception occurred.
        """
        try:
            response = requests.request(method, url, params=params, data=data, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
