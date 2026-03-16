import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# Import the market-fetching tools
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.markets import get_binary_markets
from prediction_market_agent_tooling.markets.market_type import MarketType

# 1. Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_live_context(query: str) -> str:
    print(f"🔍 Researching news for: {query}...")
    search_result = tavily.search(query=query, search_depth="advanced", max_results=3)
    return "\n".join([res['content'] for res in search_result['results']])

def run_strategy():
    # Step 1: FETCH real markets from Presagio/Omen
    # We'll grab the top 5 most recent binary markets
    print("📡 Fetching active markets from Presagio...")
    markets = get_binary_markets(limit=5, market_type=MarketType.OMEN)
    
    for market in markets:
        market_id = market.id
        title = market.question
        # Market price is usually represented as the cost of a 'YES' share
        current_price = market.get_outcome_price(0) 

        print(f"\n--- Analyzing: {title} ---")
        
        # Step 2: Research with Tavily
        context = get_live_context(title)
        
        # Step 3: Ensemble Verdict
        prompt = f"MARKET: {title}\nCONTEXT: {context}\nProvide probability [[0.XX]]."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        verdict = response.choices[0].message.content
        match = re.search(r"\[\[(\d?\.\d+)\]\]", verdict)
        ai_prob = float(match.group(1)) if match else 0.5
        
        print(f"🤖 AI Confidence: {ai_prob} | 📉 Market Price: {current_price}")

        # Step 4: Execution Logic
        if ai_prob > (current_price + 0.10):
            print(f"🚀 EDGE DETECTED ({ai_prob - current_price:.2f}). Betting 1 cent...")
            try:
                # The 'market' object from get_binary_markets is ready to trade
                market.place_bet(outcome=True, amount=0.01)
                print("✅ Trade Successful!")
            except Exception as e:
                print(f"❌ Trade failed: {e}")
        else:
            print("😴 No significant edge. Moving to next market.")

if __name__ == "__main__":
    run_strategy()
