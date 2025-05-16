import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    SAFE_GUARD_AGENT_ADDRESS: str


def agents_page() -> None:
    st.markdown(
        """## List of Agents
    
On this page you can see the list of deployed agents that can be added as signers to your Safe.
            
After you add one of the agents as a signer, he will monitor your Safe and verify queued transactions as they come automatically!

After creation of any transaction in your Safe, just wait a bit to see if it's predicted to be legit or not.

Agent will also send you a message to your Safe with the result of the validation.
"""
    )

    agent_address = Config().SAFE_GUARD_AGENT_ADDRESS
    st.markdown(f"- `{agent_address}` - https://gnosisscan.io/address/{agent_address}")
