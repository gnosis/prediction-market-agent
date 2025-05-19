import fastapi
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import initialize_langfuse
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    ChainID,
    ValidationConclusion,
    validate_safe_transaction,
)


class Config(APIKeys):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 1
    RELOAD: bool = True


def create_app() -> fastapi.FastAPI:
    nest_asyncio.apply()
    config = Config()
    initialize_langfuse(config.default_enable_langfuse)

    app = fastapi.FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping/")
    def _ping() -> str:
        return "pong"

    @app.get("/validate-transaction/")
    def _validate_transaction(
        transaction_id: str, chain_id: ChainID
    ) -> ValidationConclusion:
        logger.info(f"Validating transaction with id `{transaction_id}`.")
        result = validate_safe_transaction(
            transaction_id,
            # Can't do any of this via API, where it's expected that we aren't the signer.
            do_sign_or_execution=False,
            do_reject=False,
            do_message=False,
            chain_id=chain_id,
        )
        return result

    return app


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "prediction_market_agent.agents.safe_guard_agent.api:create_app",
        factory=True,
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=config.RELOAD,
        log_level="error",
        loop="asyncio",
    )
