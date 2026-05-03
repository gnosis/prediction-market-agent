# Fusion Agent

## Overview

The Fusion Agent is a hybrid prediction-market trading agent that combines three sources of information:

1. **Market-implied probabilities**
2. **Machine learning predictions**
3. **LLM-based contextual adjustments**

The goal is to create a more robust and calibrated forecast by blending structured data, historical patterns, and real-time information.

---

## Strategy

The agent follows a multi-stage pipeline:

### 1. Market Calibration

The raw market probability (`p_market`) is adjusted using a learned calibration model:

* Corrects for systematic biases (e.g., overpricing of YES outcomes)
* Produces a calibrated probability (`p_cal`)

---

### 2. Machine Learning Baseline

A trained ML model predicts the probability of YES using structured features:

* Market probability
* Calibrated probability
* Volume
* Duration
* Category

This produces a second estimate (`p_ml`).

---

### 3. LLM Overlay

An LLM reviews live context (via Tavily) and suggests a **small adjustment**:

* Range limited to ±0.05
* Scaled by LLM confidence
* Can skip low-quality or ambiguous markets

---

### 4. Final Prediction

The final probability is computed as:

* Base: 50% calibrated + 50% ML
* Plus: small confidence-weighted LLM adjustment

```
p_final = 0.5 * p_cal + 0.5 * p_ml + (LLM_adjustment × confidence)
```

---

## Decision Logic

A trade is placed only if:

* Edge threshold is met:

  ```
  |p_final - p_market| ≥ EDGE_THRESHOLD
  ```

* LLM does not flag the market as low-quality

* Confidence exceeds minimum threshold

---

## Key Parameters

* EDGE_THRESHOLD = 0.05
* MIN_CONFIDENCE = 0.50
* MAX_LLM_ADJUSTMENT = 0.05
* bet_on_n_markets_per_run = 3

---

## Strengths

* Combines structured + unstructured data
* More stable than pure LLM approaches
* Adaptable to real-time information
* Explicit calibration reduces market bias

---

## Weaknesses

* More complex (multiple components)
* Dependent on ML model quality
* LLM adds noise if context is weak
* Higher computational cost

---

## Notes

This agent is designed as a **balanced, general-purpose predictor** and serves as the most “complete” modeling approach in the system.

