"""
STRATEGY: The Profitability Gatekeeper
--------------------------------------
PURPOSE:
This script filters raw Omen market data to identify "High-Alpha" opportunities.
It acts as a financial filter to ensure we only spend LLM API fees on markets 
where the potential payout justifies the cost of the calculation.

INPUT: 
- scripts/historical_omen_data.csv (Raw blockchain data)

OUTPUT:
- scripts/analyzed_markets.csv (Markets labeled with 🟢 BET or 🔴 SKIP)

LOGIC:
1. Normalizes Volume: Converts 'Wei' strings into readable xDai values.
2. Calculates Market Prob: Extracts implied probability from outcome prices.
3. Computes Expected Value (EV): 
   Calculates (Potential Win * AI Probability) - (Stake * AI Failure) - API Fees.
4. Thresholding: Only marks a market as a 'BET' if the net profit is > 0 
   after accounting for the $0.01 'Brain Tax' (LLM cost).
"""


import pandas as pd
import ast

def analyze_profitability(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # 1. Constants
        STAKE = 0.10  
        LLM_COST = 0.01 
        
        # 2. Fix the Volume (Convert from string to number)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['volume_xdai'] = df['volume'] / 1e18

        # 3. Fix the Odds/Market Prob
        # This handles the [0.5, 0.5] list format that Subgrounds often returns
        def parse_odds(x):
            try:
                if isinstance(x, str):
                    val = ast.literal_eval(x)
                    return float(val[0])
                return float(x)
            except:
                return 0.5

        df['market_prob'] = df['odds'].apply(parse_odds)
        
        # 4. Simulated AI Edge
        df['ai_prob'] = (df['market_prob'] + 0.10).clip(0, 0.99)

        # 5. Calculate EV
        # Handle cases where market_prob is 0 to avoid division error
        df['potential_win'] = STAKE * (1 - df['market_prob']) / df['market_prob'].replace(0, 0.0001)
        df['expected_profit'] = (df['potential_win'] * df['ai_prob']) - (STAKE * (1 - df['ai_prob'])) - LLM_COST

        df['verdict'] = df['expected_profit'].apply(lambda x: "🟢 BET" if x > 0 else "🔴 SKIP")

        output_file = "scripts/analyzed_markets.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Analysis Complete!")
        print(f"Total Markets analyzed: {len(df)}")
        print(f"Profitable Opportunities: {len(df[df['verdict'] == '🟢 BET'])}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    analyze_profitability("scripts/historical_omen_data.csv")
