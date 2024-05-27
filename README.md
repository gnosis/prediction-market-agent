# Prediction Market Trader Agent

A library for exploring the landscape of AI Agent frameworks, using the example application of a prediction market betting agent. The various agents interact with markets from [Manifold](https://manifold.markets/), [AIOmen](https://aiomen.eth.limo/) and [Polymarket](https://polymarket.com/).

These agents build on top of the prediction market APIs from https://github.com/gnosis/prediction-market-agent-tooling.

## Setup

Install the project dependencies with `poetry`, using Python >=3.10:

```bash
python3.10 -m pip install poetry
python3.10 -m poetry install
python3.10 -m poetry shell
```

Create a `.env` file in the root of the repo with the following variables:

```bash
MANIFOLD_API_KEY=...
BET_FROM_PRIVATE_KEY=...
OPENAI_API_KEY=...
```

Depending on the agent you want to run, you may require additional variables. See an exhaustive list in `.env.example`.

## Interactive Streamlit Apps

- An autonomous agent with function calling. Can be 'prodded' by the user to guide its strategy: `streamlit run prediction_market_agent/agents/microchain_agent/app.py` (Deployed [here](https://autonomous-trader-agent.streamlit.app/))
- Pick a prediction market question, or create your own, and pick one or more agents to perform research and make a prediction: `streamlit run scripts/agent_app.py` (Deployed [here](https://prediction-market-agent-tooling-monitor.streamlit.app))

## Dune Dashboard

The on-chain activity of the deployed agents from this repo can be tracked on a Dune dashboard [here](https://dune.com/hdser/omen-ai-agents).

## Running (*deprecated* see https://github.com/gnosis/prediction-market-agent/issues/211)

Execute `main.py` with optional arguments:

```bash
% python main.py --help
 Usage: main.py [OPTIONS]                                                                    

 Picks one market and answers it, optionally placing a bet.                                  

╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --market-type                     [manifold|omen|polymarket]  [default: manifold]         │
│ --agent-type                      [langchain|autogen|always_  [default: always_yes]       │
│                                   yes|llamaindex|metagpt|cre                              │
│                                   wai|custom_openai|custom_l                              │
│                                   lama]                                                   │
│ --auto-bet       --no-auto-bet                                [default: no-auto-bet]      │
│ --help                                                        Show this message and exit. |
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

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

## Contributing

See the [Issues](https://github.com/gnosis/prediction-market-agent/issues) for ideas of things that need fixing or implementing. The team is also receptive to new issues and PRs.

An great self-contained first contribution would be to implement an agent using a framework in the ['Other frameworks to try'](https://github.com/gnosis/prediction-market-agent/issues/210) issue.
