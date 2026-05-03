# Multi-Persona Ensemble Agent

## Overview

The Multi-Persona Ensemble Agent is an LLM-based forecasting system that generates multiple independent predictions using different reasoning “personas” and aggregates them into a final probability.

The goal is to improve forecast robustness by combining diverse perspectives rather than relying on a single model or reasoning path.

---

## Strategy

The agent simulates four distinct personas:

* **Researcher** → focuses on current evidence and news
* **Skeptic** → challenges assumptions and looks for failure cases
* **Trader** → compares expected reality to current market pricing
* **Risk Manager** → evaluates uncertainty and whether the market should be skipped

Each persona produces:

* Probability estimate
* Confidence score
* Skip recommendation
* Short reasoning

---

## Aggregation

The final probability is calculated as a weighted average:

* Researcher: 35%
* Trader: 25%
* Skeptic: 20%
* Risk Manager: 20%

The agent also computes disagreement:

```python
disagreement = max(persona_probs) - min(persona_probs)
```

---

## Decision Logic

A trade is placed only if all conditions are met:

### 1. No risk veto

At least two personas must not vote to skip.

---

### 2. Low disagreement

```python
disagreement ≤ MAX_DISAGREEMENT
```

---

### 3. Sufficient edge

```python
edge = |final_prob - market_prob| ≥ EDGE_THRESHOLD
```

---

## Trade Direction

The agent can trade both sides:

```python
trade_side = "YES" if final_prob > market_prob else "NO"
```

* If model probability > market → bet YES
* If model probability < market → bet NO

---

## Key Parameters

* EDGE_THRESHOLD = 0.07
* MAX_DISAGREEMENT = 0.25
* bet_on_n_markets_per_run = 3

Note: `MIN_PROFIT_RATIO` is defined in the code but not currently used in filtering.

---

## Betting Strategy

Uses Kelly-scaled bet sizing with bounded position sizes

This keeps risk controlled while allowing larger bets when edge is stronger.

---

## Logging

Forecasts are stored in:

```text
forecast_log.csv
```

Tracked fields include:

* market probability
* final probability
* confidence
* disagreement
* trade decision
* trade side

---

## Strengths

* Combines multiple reasoning perspectives
* Uses real-time context via Tavily
* Balanced strategy (can bet YES or NO)
* Built-in disagreement and skip logic
* More robust than single-model approaches

---

## Weaknesses

* Expensive (multiple LLM calls per market)
* Sensitive to prompt quality and search results
* Can be influenced by noisy or incomplete context
* More complex than simpler baseline agents

---

## Notes

This agent serves as the **primary general-purpose predictor** in the system.

This strategy is more flexible and aims to balance accuracy and profitability by identifying opportunities on both sides of the market.

