import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Initialize the new Client - it automatically looks for GOOGLE_API_KEY in .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_ensemble_prediction(market_title):
    print(f"Analyzing: {market_title}")
    model_id = "gemini-2.0-flash" # Use the latest 2026 model
    
    # Step 1: The Arguments
    bull_reasoning = client.models.generate_content(
        model=model_id, 
        contents=f"Argue why 'YES' is the most likely outcome for: {market_title}. Focus on news and trends."
    ).text
    
    bear_reasoning = client.models.generate_content(
        model=model_id, 
        contents=f"Argue why 'NO' is the most likely outcome for: {market_title}. Focus on risks and delays."
    ).text
    
    # Step 2: The Consensus Judge
    judge_prompt = f"""
    Market: {market_title}
    Argument for YES: {bull_reasoning}
    Argument for NO: {bear_reasoning}
    
    Acting as a professional prediction market analyst, weigh these views. 
    Provide a final probability (0.0 to 1.0) that the outcome will be 'YES'.
    Return ONLY the number.
    """
    
    verdict = client.models.generate_content(model=model_id, contents=judge_prompt).text.strip()
    return float(verdict)

if __name__ == "__main__":
    test_title = "Will Mercedes-Benz publicly announce the start of Level 3 autonomy in the US by end of 2026?"
    try:
        prob = get_ensemble_prediction(test_title)
        print(f"✅ Ensemble Probability: {prob}")
    except Exception as e:
        print(f"❌ Error: {e}")
