# Prediction Market Betting Agent

A library for building an autonomous agent that researches markets from [Manifold](https://manifold.markets/), and uses its findings to place bets.

## Setup

Install requirements in a virtual environment

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

Create a `.env` file in the base directory that contains the following environmnent variables:

```bash
MANIFOLD_API_KEY=...
SERP_API_KEY=...
OPENAI_API_KEY=...
```

## TODOs

- Extend the agent with tools to pick and bet on the market.
