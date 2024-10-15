import typer

from prediction_market_agent.db.pinecone_handler import PineconeHandler


def main() -> None:
    """Script for inserting all markets into Pinecone (if not yet there)."""
    PineconeHandler().insert_open_omen_markets_if_not_exists()


if __name__ == "__main__":
    typer.run(main)
