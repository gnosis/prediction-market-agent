# Prediction Market Betting Agent

A library for exploring the landscape of AI Agent frameworks, using the example application of a prediction market betting agent. The agent researches markets from [Manifold](https://manifold.markets/), and uses its findings to place bets.

## Setup

Use Poetry to manage environment:

```
pip install poetry
poetry install
```

And either execute into the environment with `poetry shell` or use commands as `poetry run python main.py ...`.

Optionally, one can use standard virtual environment without Poetry:

```bash
python3 -m venv .venv
source .venv/bin/activate
poetry export -f requirements.txt --output requirements.txt
pip install -f requirements.txt
```

Create a `.env` file in the base directory that contains the following environmnent variables:

```bash
MANIFOLD_API_KEY=...
SERP_API_KEY=...
OPENAI_API_KEY=...
REPLICATE_API_TOKEN=...
```

## Running

Execute `main.py` with optional arguments:

```bash
% python main.py --help
Usage: main.py [OPTIONS]

  Picks one market and answers it, optionally placing a bet.

Options:
  --market-type [manifold|omen]   [default: MarketType.MANIFOLD]
  --agent-type [langchain|autogen|always_yes|coin_flip|llamaindex|metagpt|crewai|custom_openai|custom_llama]
                                  [default: AgentType.ALWAYS_YES]
  --auto-bet / --no-auto-bet      [default: no-auto-bet]
  --help                          Show this message and exit.
```

## Testing

Run the type checking:

```bash
mypy
```

Run the tets:

```bash
pytest tests
```

Note: these make actual API calls!

## Deploying

To deploy an agent to google cloud, see the example in `examples/cloud_deployment`.

Requires the gcloud cli (see [here](https://cloud.google.com/sdk/docs/install)), and auth to have been completed (i.e. `gcloud auth login`).

## Frameworks implemented

| Framework | Notes |
| --------- | ---------------- |
| [LangChain](https://python.langchain.com/docs/modules/agents/) | Wraps OpenAI function calling API. Equips single agent with tools, makes calls in while loop. Has library of built-in tools, or can make own. Can extend using [Python Repl](https://python.langchain.com/docs/integrations/tools/python) tool to execute ad-hoc generated code. |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/use_cases/agents.html) | Similar to LangChain. Tool library from `llama_hub`. No tool to execute ad-hoc generated code. |
| [MetaGPT](https://github.com/geekan/MetaGPT) | Advertised as a framework for building an agent-based software dev team, but can be general purpose. Can do single-agent or multi-agent (as a [Team](https://docs.deepwisdom.ai/main/en/guide/tutorials/multi_agent_101.html)) execution. Has many predefined agent `Role`s, pre-equipped with relevant tools. |
| [AutoGen](https://github.com/microsoft/autogen) | Multi-agent. Can use local model. Easy to explicitly control the execution pattern of the agents. `GPTAssistantAgent` class wraps the OpenAI Assistant API. |
| [crewAI](https://github.com/joaomdmoura/crewAI) | Similar to AutoGen, except agents<->task mapping is only 1-1 currently. Agents tackly distinct tasks in series. Can use local model (Ollama integration). Agent tool integration with LangChain, so can use its existing tool library. |

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
- [OpenAI assistants API](https://platform.openai.com/docs/assistants)
- [SuperAGI](https://github.com/TransformerOptimus/SuperAGI)
- [Open Interpreter](https://github.com/KillianLucas/open-interpreter)
- [Agent OS](https://github.com/smartcomputer-ai/agent-os)
- [Council](https://github.com/chain-ml/council)

## TODOs

- Implement agents for frameworks in `Other frameworks to try`
- Add option to main.py to use a locally running llm (e.g. Ollama+litellm).
- Extend the agent with tools to pick the market.
- Split agent functionality into:
  1. Researcher: generates report from a market question
  2. Predictor: Generates a `p_yes`, `p_no` for the market, from which it places a bet.
- Seeing some reponses like `Error: 401 Client Error: HTTP Forbidden for url` from web scraping tool (e.g. for Reuters). Look to improve.
- Add `polymarket.py` that contains abstractions for getting markets and placing bets with the [Polymarket prediction market](https://polymarket.com/).
