# Prediction Market Betting Agent

A library for exploring the landscape of AI Agent frameworks, using the example application of a prediction market betting agent. The agent researches markets from [Manifold](https://manifold.markets/), and uses its findings to place bets.

## Setup

Install requirements in a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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

## Testing

Run the tets:

```bash
pytest tests
```

Note: these make actual API calls!

## Frameworks implemented

| Framework | Implemented Yet? |
| --------- | ---------------- |
| [LangChain](https://python.langchain.com/docs/modules/agents/) | Yes |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/use_cases/agents.html) | Yes |
| [MetaGPT](https://github.com/geekan/MetaGPT) | Yes |
| [AutoGen](https://github.com/microsoft/autogen) | Yes |
| [crewAI](https://github.com/joaomdmoura/crewAI) | Yes |
| [OpenAI assistants API](https://platform.openai.com/docs/assistants) | No |

### Stale projects

A list of framework projects that had traction but are no longer under development.

- [RoboGPT](https://github.com/rokstrnisa/RoboGPT)
  - Note: doesn't specify an `openai` version in its dependencies, and is incompatible with the latest version.
- [BabyAGI](https://github.com/yoheinakajima/babyagi)

### Other frameworks to try

- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Tavily GPT Researcher](https://github.com/assafelovic/gpt-researcher)
  - Note: Pip installing this [package](https://docs.tavily.com/docs/gpt-researcher/pip-package) breaks langchain agent atm due to incompatible dependencies.

## TODOs

- Add `omen.py` that contains abstractions for getting markets and placing bets with the [Omen prediction market](https://omen.eth.limo/).
- Implement agents for frameworks in `Other frameworks to try`
- Extend the agent with tools to pick and bet on the market.
- Bet more intelligently, based on prediction probability, market odds, market liquidity etc.
- Split agent functionality into:
  1. Researcher: generates report from a market question
  2. Predictor: Generates a `p_yes`, `p_no` for the market, from which it places a bet.
- Seeing some reponses like `Error: 401 Client Error: HTTP Forbidden for url` from web scraping tool (e.g. for Reuters). Look to improve.
