import threading
import time
from types import TracebackType
from typing import Callable, Optional, Type

import uvicorn
from fastapi import FastAPI, Request, Response


class UvicornServer:
    """
    Util class to run a FastAPI app in a separate thread, to enable testing
    of fastapi endpoints.
    """

    def __init__(self, app: FastAPI, host: str = "localhost", port: int = 8000) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "UvicornServer":
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.5)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.server.should_exit = True
        self.thread.join()


def to_fastapi_app(fn: Callable[..., Response]) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def endpoint(request: Request) -> Response:
        query_params = dict(request.query_params)
        return fn(**query_params)

    return app
