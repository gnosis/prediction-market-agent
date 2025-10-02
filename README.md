# Gnosis Agent

A library for exploring the landscape of AI Agent frameworks, using the example application of a prediction market betting agent. The various agents interact with markets from [Manifold](https://manifold.markets/), [Presagio](https://presagio.pages.dev/) and [Polymarket](https://polymarket.com/).

These agents build on top of the prediction market APIs from https://github.com/gnosis/prediction-market-agent-tooling.

## Setup

Install the project dependencies with `poetry`, using Python >=3.11:

```bash
python3.11 -m pip install poetry
python3.11 -m poetry install
python3.11 -m poetry shell
```

Create a `.env` file in the root of the repo with the following variables:

```bash
MANIFOLD_API_KEY=...
BET_FROM_PRIVATE_KEY=...
OPENAI_API_KEY=...
GRAPH_API_KEY=...
```

**Note:** The `GRAPH_API_KEY` is required for agents that interact with Omen markets. You can obtain one for free from [The Graph](https://thegraph.com).

**Important:** When running agents on Omen (Gnosis Chain), your wallet address (derived from `BET_FROM_PRIVATE_KEY`) must have some xDai to pay for transaction fees. Without xDai, you'll encounter a `CantPayForGasError`. You can either:
- Send xDai to your wallet address, or
- Set `GNOSIS_RPC_URL` environment variable to a local chain (e.g., `anvil`) where you have test funds.

Depending on the agent you want to run, you may require additional variables. See an exhaustive list in `.env.example`.

## Interactive Streamlit Apps

- An autonomous agent with function calling. Can be 'prodded' by the user to guide its strategy: `streamlit run prediction_market_agent/agents/microchain_agent/app.py` (Deployed [here](https://autonomous-trader-agent.ai.gnosisdev.com))
- Pick a prediction market question, or create your own, and pick one or more agents to perform research and make a prediction: `streamlit run scripts/agent_app.py` (Deployed [here](https://pma-agent.ai.gnosisdev.com))

## Dune Dashboard

The on-chain activity of the deployed agents from this repo can be tracked on a Dune dashboard [here](https://dune.com/gnosischain_team/omen-ai-agents).

## Running

Execute `prediction_market_agent/run_agent.py`, specifying the ID of the 'runnable agent', and the market type as arguments:

```bash
% python prediction_market_agent/run_agent.py --help

 Usage: run_agent.py [OPTIONS] AGENT:{coinflip|replicate_to_omen|think_thorough                                         
                     ly|think_thoroughly_prophet|think_thoroughly_prophet_kelly                                         
                     |knownoutcome|microchain|microchain_modifiable_system_prom                                         
                     pt_0|microchain_modifiable_system_prompt_1|microchain_modi                                         
                     fiable_system_prompt_2|microchain_modifiable_system_prompt                                         
                     _3|microchain_with_goal_manager_agent_0|metaculus_bot_tour                                         
                     nament_agent|prophet_gpt4o|prophet_gpt4|prophet_gpt4_final                                         
                     |prophet_gpt4_kelly|olas_embedding_oa|social_media|omen_cl                                         
                     eaner|ofv_challenger}                                                                              
                     MARKET_TYPE:{omen|manifold|polymarket|metaculus}                                                   
                                                                                                                        
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    agent            AGENT:{coinflip|replicate_to_omen|think_thorou  [default: None] [required]                     │
│                       ghly|think_thoroughly_prophet|think_thoroughly                                                 │
│                       _prophet_kelly|knownoutcome|microchain|microch                                                 │
│                       ain_modifiable_system_prompt_0|microchain_modi                                                 │
│                       fiable_system_prompt_1|microchain_modifiable_s                                                 │
│                       ystem_prompt_2|microchain_modifiable_system_pr                                                 │
│                       ompt_3|microchain_with_goal_manager_agent_0|me                                                 │
│                       taculus_bot_tournament_agent|prophet_gpt4o|pro                                                 │
│                       phet_gpt4|prophet_gpt4_final|prophet_gpt4_kell                                                 │
│                       y|olas_embedding_oa|social_media|omen_cleaner|                                                 │
│                       ofv_challenger}                                                                                │
│ *    market_type      MARKET_TYPE:{omen|manifold|polymarket|metaculu  [default: None] [required]                     │
│                       s}                                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                              │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.       │
│ --help                        Show this message and exit.                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Troubleshooting

### Testing Agent Prediction Logic Without Full Setup

If you want to test just the prediction logic of an agent without needing all the environment variables or xDai funds, you can do this:

```python
from prediction_market_agent.agents.coinflip_agent.deploy import DeployableCoinFlipAgent
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket, SortBy

agent = DeployableCoinFlipAgent()
markets = OmenAgentMarket.get_binary_markets(limit=1, sort_by=SortBy.CLOSING_SOONEST)
prediction = agent.answer_binary_market(markets[0])
```

This will skip any blockchain transactions and balance checks, allowing you to test the prediction part only. You can also initialize `OmenAgentMarket` manually with your own question to avoid needing the `GRAPH_API_KEY`.

### Common Errors

**`CantPayForGasError`**: Your wallet (from `BET_FROM_PRIVATE_KEY`) has insufficient xDai on Gnosis Chain to pay transaction fees. Either fund your wallet or use a local testnet.

**Missing API Key errors**: Different agents require different API keys. Check `.env.example` for the full list and ensure you have the keys for the specific agent you're trying to run.

## Deploying

The easiest way to make your own agent that places a bet on a prediction market is to subclass the `DeployableTraderAgent`. See `DeployableCoinFlipAgent` for a minimal example.

From there, you can add it to the `RUNNABLE_AGENTS` dict in `prediction_market_agent/run_agent.py`, and use that as the entrypoint for running the agent in your cloud deployment.

## Contributing

See the [Issues](https://github.com/gnosis/prediction-market-agent/issues) for ideas of things that need fixing or implementing. The team is also receptive to new issues and PRs.

A great self-contained first contribution would be to implement an agent using a framework in the ['Other frameworks to try'](https://github.com/gnosis/prediction-market-agent/issues/210) issue.
