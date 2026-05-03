# NoMatic Agent

## Overview

The NoMatic Agent is a contrarian trading strategy designed to exploit systematic overpricing of YES outcomes in prediction markets.

It leverages historical base rates to identify when markets are overly optimistic and selectively places NO bets.

---

## Strategy

The core idea:

> Markets tend to overestimate the likelihood of YES outcomes.

The agent compares:

* Current market probability (`p_yes`)
* Historical YES rate from recently resolved markets

---

## Decision Logic

A trade is placed only if all conditions are met:

### 1. Market is NO-dominant historically

```
recent_no_rate ≥ MIN_RECENT_NO_RATE
```

---

### 2. Market is pricing YES highly

```
p_yes ≥ MIN_MARKET_PYES
```

---

### 3. Overpricing edge exists

```
edge = p_yes - recent_yes_rate ≥ MIN_EDGE
```

---

## Probability Adjustment

The agent computes an adjusted probability:

```
adjusted_p_yes = 
    BASE_RATE_WEIGHT × recent_yes_rate 
  + (1 - BASE_RATE_WEIGHT) × market_prob
```

Then caps it:

```
adjusted_p_yes ≤ MAX_ADJUSTED_PYES
```

This ensures the agent maintains a **NO bias**.

---

## Key Parameters

* LOOKBACK_DAYS = 30
* MIN_MARKET_PYES = 0.60
* MIN_EDGE = 0.20
* BASE_RATE_WEIGHT = 0.65
* MAX_ADJUSTED_PYES = 0.45
* MIN_RECENT_NO_RATE = 0.55

---

## Strengths

* Exploits systematic market inefficiencies
* High ROI potential on mispriced markets
* Simple and interpretable logic
* Low computational cost

---

## Weaknesses

* One-sided strategy (mostly bets NO)
* High variance outcomes
* Can miss genuine high-probability YES events
* Sensitive to base rate assumptions

---

## Notes

This agent is intentionally **directional and contrarian**.

It represents a “high-risk, high-reward” component of the overall agent portfolio.

