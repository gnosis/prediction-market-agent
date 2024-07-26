import requests
from fastapi import Response

from tests.endpoints.utils import UvicornServer, to_fastapi_app


def hello_world(name: str = "World") -> Response:
    return Response(content=f"Hello {name}")


def test_hello_world() -> None:
    with UvicornServer(to_fastapi_app(hello_world)) as server:
        for name in ["foo", None]:
            response = requests.get(server.url, params={"name": name})
            response.raise_for_status()
            name = "World" if name is None else name
            assert response.text == f"Hello {name}"
