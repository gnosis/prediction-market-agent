# Fusion Agent

## Overview

The Fusion Agent is a hybrid forecasting and trading agent that combines calibrated market probabilities, machine learning predictions, and LLM-based contextual reasoning into a single prediction pipeline.

Unlike the purely LLM-driven agents or the rule-based contrarian strategy, the Fusion Agent is designed to balance statistical structure with real-time information.

---

## Strategy

The agent combines three forecasting layers:

### 1. Market Calibration

The raw market-implied probability (`p_market`) is passed through a learned calibration model:

```python
p_cal = calibration_model.predict(p_market)
```

This attempts to correct systematic market biases identified during historical analysis.

---

### 2. Machine Learning Baseline

A supervised ML model generates an independent probability estimate using structured market features:

* Market implied probability
* Calibrated probability
* Volume
* Market duration
* Market category

This produces:

```python
p_ml
```

which serves as a statistical baseline forecast.

---

### 3. LLM Context Overlay

The agent retrieves live context using Tavily and asks an LLM to suggest a **small bounded adjustment** to the baseline forecast.

The LLM:

* reviews current information
* evaluates whether evidence supports moving probability up or down
* can recommend skipping weak or ambiguous markets

The adjustment is intentionally capped:

```python
MAX_LLM_ADJUSTMENT = 0.05
```

to prevent the LLM from dominating the prediction.

---

## Final Prediction

The calibrated and ML probabilities are blended:

```python
p_base = 0.5 * p_cal + 0.5 * p_ml
```

Then a confidence-scaled LLM adjustment is added:

```python
p_final = p_base + (llm_adjustment × confidence)
```

This creates a stable prediction backbone with a smaller real-time overlay.

---

## Decision Logic

A trade is placed only if:

### 1. LLM does not recommend skipping

Markets with stale, weak, or noisy evidence are filtered out.

---

### 2. Confidence exceeds threshold

```python
confidence ≥ MIN_CONFIDENCE
```

---

### 3. Edge exceeds threshold

```python
edge = |p_final - p_market| ≥ EDGE_THRESHOLD
```

---

## Trade Direction

The Fusion Agent can place both YES and NO trades:

* If `p_final > p_market` → bet YES
* If `p_final < p_market` → bet NO

This makes it a balanced forecasting strategy rather than a directional bias system.

---

## Key Parameters

* EDGE_THRESHOLD = 0.05
* MIN_CONFIDENCE = 0.50
* MAX_LLM_ADJUSTMENT = 0.05
* bet_on_n_markets_per_run = 3

---

## Betting Strategy

Uses Kelly-scaled position sizing with bounded trade sizes:

* Small wallet testing:

  * min bet: $0.03
  * max bet: $0.08

This allows scaling exposure while maintaining controlled risk.

---

## Logging

Forecasts are stored in:

```text
forecast_log.csv
```

Tracked fields include:

* raw market probability
* calibrated probability
* ML prediction
* LLM adjustment
* final probability
* confidence
* trade decision
* skip reason

---

## Strengths

* Combines statistical modeling with live contextual reasoning
* More stable than pure LLM-based forecasting
* Explicit calibration layer reduces market bias
* Uses structured ML features alongside real-time information
* Flexible enough to identify both YES and NO opportunities

---

## Notes

The Fusion Agent serves as the primary **hybrid forecasting model** in the project.

It was designed to balance:

* predictive accuracy
* calibration
* adaptability
* and profitability under live market conditions.
