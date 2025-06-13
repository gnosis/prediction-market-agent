from pydantic_settings import BaseSettings, SettingsConfigDict


class PostHogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    POSTHOG_API_KEY: str | None = None
