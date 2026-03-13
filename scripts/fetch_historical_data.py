import os
import pandas as pd
from subgrounds import Subgrounds
from dotenv import load_dotenv

load_dotenv()

def fetch_data():
    sg = Subgrounds()
    api_key = os.getenv("GRAPH_API_KEY")
    
    if not api_key:
        print("Error: GRAPH_API_KEY not found in .env")
        return

    # Verified 2026 Gnosis Omen Subgraph ID
    subgraph_id = "9fUVQpFwzpdWS9bq5WkAnmKbNNcoBwatMR4yZq81pbbz"
    omen_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"
    
    try:
        omen = sg.load_subgraph(omen_url)

        # We pull the last 100 markets created
        # We removed the 'isResolved' filter to avoid the schema error
        markets = omen.Query.fixedProductMarketMakers(
            first=100,
            orderBy=omen.FixedProductMarketMaker.creationTimestamp,
            orderDirection='desc'
        )

        df = sg.query_df([
            markets.question.title,
            markets.category,
            markets.collateralVolume,
            markets.outcomeTokenMarginalPrices,
            markets.creationTimestamp
        ])

        # Clean up column names for easier ML use later
        df.columns = ['title', 'category', 'volume', 'odds', 'created_at']

        # Save to your scripts folder
        df.to_csv("scripts/historical_omen_data.csv", index=False)
        print(f"Success! Saved {len(df)} markets to scripts/historical_omen_data.csv")
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    fetch_data()
