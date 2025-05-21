from typing import Callable

import streamlit as st
import streamlit.components.v1 as components
from prediction_market_agent_tooling.gtypes import ChainID
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    SAFE_GUARD_API_URL: str


def get_api_page(chain_id: ChainID) -> Callable[[], None]:
    def api_page() -> None:
        safe_guard_api_url = Config().SAFE_GUARD_API_URL

        st.markdown(
            f"""## API Documentation


On this page, you can see documentation for the Safe Guard API available at {safe_guard_api_url}.            


You can use this in your applications to verify your transactions, before you sign them.
"""
        )

        components.iframe(safe_guard_api_url, height=750, scrolling=True)

    return api_page
