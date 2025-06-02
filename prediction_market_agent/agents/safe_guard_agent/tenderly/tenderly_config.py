import typing as t

from prediction_market_agent_tooling.gtypes import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TenderlyKeys(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    TENDERLY_ACCOUNT_SLUG: t.Optional[str] = None
    TENDERLY_PROJECT_SLUG: t.Optional[str] = None
    TENDERLY_ACCESS_KEY: t.Optional[SecretStr] = None
