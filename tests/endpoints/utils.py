import threading
import time
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request


class UvicornServer:
    """
    Util class to run a FastAPI app in a separate thread, to enable testing
    of fastapi endpoints.
    """

    def __init__(self, app, host="localhost", port=8000):
        self.app = app
        self.host = host
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server.should_exit = True
        self.thread.join()


def to_fastapi_app(fn: Callable) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def endpoint(request: Request):
        query_params = dict(request.query_params)
        return fn(**query_params)

    return app
