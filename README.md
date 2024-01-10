# Prediction Market Betting Agent

A library for exploring the landscape of AI Agent frameworks, using the example application of a prediction market betting agent. The agent researches markets from [Manifold](https://manifold.markets/), and uses its findings to place bets.

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

## Running

Execute `main.py` with optional arguments:

```bash
% python main.py --help
usage: main.py [-h] [--agent-type {langchain,autogen,always_yes}] [--auto-bet AUTO_BET]

optional arguments:
  -h, --help            show this help message and exit
  --agent-type {langchain,autogen,always_yes}
  --auto-bet AUTO_BET   If true, does not require user input to place the bet.
```

## Frameworks implemented

| Framework | Implemented Yet? |
| --------- | ---------------- |
| LangChain |	Yes |
| LlamaIndex | No |
| MetaGPT | No |
| AutoGen | Yes |
| crewAI | No |
| BabyAGI | No |
| RoboGPT | No |
| OpenAI assistants API | No |
| Tavily GPT Researcher | No |

## Other TODOs

- Extend the agent with tools to pick and bet on the market.
- Bet more intelligently, based on prediction probability, market odds, market liquidity etc.
- Split agent functionality into:
  1. Researcher: generates report from a market question
  2. Predictor: Generates a `p_yes`, `p_no` for the market, from which it places a bet.
