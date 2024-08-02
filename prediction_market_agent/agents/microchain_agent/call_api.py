import json
from typing import Optional

import requests
from microchain import Function

from prediction_market_agent.utils import APIKeys


class CallAPI(Function):
    @property
    def description(self) -> str:
        return (
            "Use this function to make arbitrary API calls to any endpoint with any parameters. "
            "It supports different HTTP methods (GET, POST, etc.) and can include headers, "
            "query parameters, and request bodies. "
            "Params, data and headers needs to be passed as JSON strings."
        )

    @property
    def example_args(self) -> list[str]:
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


class SendTelegramMessage(Function):
    @property
    def description(self) -> str:
        return "Use this function to send message on Telegram."

    @property
    def example_args(self) -> list[str]:
        return ["123", "Hello from your telegram bot!"]

    def __call__(
        self,
        chat_id: str,
        message: str,
    ) -> str:
        url = f"https://api.telegram.org/bot{APIKeys().telegram_bot_key.get_secret_value()}/sendMessage?chat_id={chat_id}&text={message}"
        response = requests.get(url)
        response.raise_for_status()
        return "Message sent"


class GetTasks(Function):
    @property
    def description(self) -> str:
        return "Use this function to get available tasks in a JSON dumped format."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return json.dumps(
            [
                {
                    "id": "1",
                    "title": """Solve the following math problem:

If there are 3 apples and you take away 2, how many do you have?
                    
Send the answer to Telegram Chat ID 6514789173.
""",
                    "reward": {
                        "amount": 1,
                        "currency": "wxDai",
                    },
                },
                {
                    "id": "2",
                    "title": """
Do a prediction on the following Omen market with id `0xa6febd6ffda06af320f5d92ec3dd562149cc2bc7`.                  

Send the result to Telegram Chat ID 6514789173.
""",
                    "reward": {
                        "amount": 1,
                        "currency": "wETH",
                    },
                },
                {
                    "id": "3",
                    "title": "Install Python library `chihuahua` and then execute the following code: `import chihuahua; print(42)`.",
                    "reward": {
                        "amount": 2,
                        "currency": "wxDai",
                    },
                },
            ],
            indent=2,
        )


API_FUNCTIONS: list[type[Function]] = [
    CallAPI,
    SendTelegramMessage,
    GetTasks,
]
