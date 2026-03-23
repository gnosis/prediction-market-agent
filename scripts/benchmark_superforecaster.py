import numpy as np
from unittest.mock import MagicMock
from dotenv import load_dotenv

load_dotenv()

from prediction_market_agent.agents.superforecaster_agent.deploy import SuperforecasterAgent

# ── questions from the benchmark report ──────────────────────────
questions = [
    "Will Iran shoot down a US military plane/helicopter by end of March?",
    "Will I deem Donut Lab's solid state battery claims in their video to have been false one year from now?",
    "Will the US put boots on the ground in Iran in 2026?",
    "Are there more forks than spoons in the world?",
    "Will USA change Iran regime",
    "Will Chuck Norris resurrect from the dead on the third day? (Saturday March 21st)",
    "Will BTC exceed 73k before Sunday, March 22, 2026?",
    "Will MiniMax release the weights of its M2.7 model by the end of the week?",
    "Will Iran launch the 100th wave of missile/drone strike under 'Operation True Promise 4' before March 27?",
    "Is this jigsaw puzzle missing some piece?",
]

# ── reference probs from the benchmark report ─────────────────────
reference = {
    "Will Iran shoot down a US military plane/helicopter by end of March?": 0.378,
    "Will I deem Donut Lab's solid state battery claims in their video to have been false one year from now?": 0.572,
    "Will the US put boots on the ground in Iran in 2026?": 0.660,
    "Are there more forks than spoons in the world?": 0.840,
    "Will USA change Iran regime": 0.169,
    "Will Chuck Norris resurrect from the dead on the third day? (Saturday March 21st)": 0.010,
    "Will BTC exceed 73k before Sunday, March 22, 2026?": 0.109,
    "Will MiniMax release the weights of its M2.7 model by the end of the week?": 0.123,
    "Will Iran launch the 100th wave of missile/drone strike under 'Operation True Promise 4' before March 27?": 0.410,
    "Is this jigsaw puzzle missing some piece?": 0.865,
}

# ── advanced agent results for comparison ─────────────────────────
advanced_agent = {
    "Will Iran shoot down a US military plane/helicopter by end of March?": 0.75,
    "Will I deem Donut Lab's solid state battery claims in their video to have been false one year from now?": 0.75,
    "Will the US put boots on the ground in Iran in 2026?": 0.65,
    "Are there more forks than spoons in the world?": 0.45,
    "Will USA change Iran regime": 0.25,
    "Will Chuck Norris resurrect from the dead on the third day? (Saturday March 21st)": 0.0,
    "Will BTC exceed 73k before Sunday, March 22, 2026?": 0.65,
    "Will MiniMax release the weights of its M2.7 model by the end of the week?": 0.25,
    "Will Iran launch the 100th wave of missile/drone strike under 'Operation True Promise 4' before March 27?": 0.25,
    "Is this jigsaw puzzle missing some piece?": 0.65,
}

def make_mock_market(question: str):
    market = MagicMock()
    market.question = question
    return market

def main():
    agent = SuperforecasterAgent()

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/10] {q[:80]}")
        print("  Running...")
        market = make_mock_market(q)

        try:
            answer = agent.answer_binary_market(market)
            p_yes = float(answer.p_yes) if answer else None
            confidence = float(answer.confidence) if answer else None
        except Exception as e:
            print(f"  ERROR: {e}")
            p_yes = None
            confidence = None

        results.append({
            "question": q,
            "p_yes": p_yes,
            "confidence": confidence,
        })

        if p_yes is not None:
            print(f"  p_yes={p_yes:.2f}  confidence={confidence:.2f}")

    # ── compute MSE ───────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Question':<45} {'Ref':>5} {'SF':>5} {'Adv':>5}")
    print("="*70)

    sf_errors = []
    adv_errors = []

    for r in results:
        q = r["question"]
        ref = reference.get(q)
        sf = r["p_yes"]
        adv = advanced_agent.get(q)

        if ref is not None and sf is not None:
            sf_errors.append((sf - ref) ** 2)
        if ref is not None and adv is not None:
            adv_errors.append((adv - ref) ** 2)

        sf_str  = f"{sf:.2f}"  if sf  is not None else "N/A"
        adv_str = f"{adv:.2f}" if adv is not None else "N/A"
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"

        print(f"{q[:45]:<45} {ref_str:>5} {sf_str:>5} {adv_str:>5}")

    print("="*70)
    if sf_errors:
        print(f"Superforecaster MSE : {np.mean(sf_errors):.4f}")
    print(f"AdvancedAgent MSE   : 1.4193")
    print(f"CoinFlip MSE        : 4.9422")

if __name__ == "__main__":
    main()