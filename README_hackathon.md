# Hackathon Quickstart Guide

This guide futher expands on how to create your first agent using the Prediction Market Agent framework. It is intended for hackathon participants or anyone who wants a quickstart to creating a new agent.

About prediction markets themselves, you can read more in <https://support.metamask.io/manage-crypto/trade/predict/what-are-prediction-markets>.

## Setup

After installing the repository as mentioned in the README.md, add the following environment variables to your `.env` file, which is copied from `.env.example`:

- GRAPH_API_KEY: can be obtained for free on <https://thegraph.com> (Required for querying the data)
- SERPER_API_KEY: can be obtained for free on <https://serper.dev> (Required for google search function)
- FIRECRAWL_API_KEY: can be obtained for free on <https://www.firecrawl.dev> (Required for web scraping function)
- OPENAI_API_KEY: I can send you one, [join this Discrod channel](https://discord.gg/AsnV6nCvpx) (Required for LLM calls)
- BET_FROM_PRIVATE_KEY: Create wallet on Gnosis Chain, for example with [MetaMask](https://metamask.io/) and I can send you some xDai (required for doing transactions on the chain)
  - By default, MetaMask doesn't have Gnosis Chain listed. You need to click on the network selection in top left, click add a new one, and fill in:
    - Name: Gnosis Chain
    - RPC URL: <https://rpc.gnosischain.com>
    - Chain ID: 100
    - Symbol: XDAI
- MANIFOLD_API_KEY: can be obtained for free on <https://manifold.markets> (Required for running benchmark)

## Run

```bash
python prediction_market_agent/run_agent.py coinflip omen
```

or

```bash
python prediction_market_agent/run_agent.py advanced_agent omen
```

or, if you create a brand new agent and add its class to `run_agent.py`, then run

```bash
python prediction_market_agent/run_agent.py your_agent omen
```

## Task

In short: Implement new logic for trading on prediction markets, in any way you deem best.

The goal is to get as good predictions as possible for as cheap as possible. (given the costs such as 3rd party services, LLM calls, etc.)

### Recommended steps

1. Take a look at `prediction_market_agent/agents/coinflip_agent/deploy.py`, this is the simplest agent possible, but with random guesses won't do any good. Try to run it and see the logs.

2. Take a closer look at `prediction_market_agent/agents/advanced_agent/deploy.py`, this is an agent that actually can predict something useful, because it's retrieving up-to-date information from websites. Try to run it and see the logs.

3. Take a around this repository, mainly into [the agents directory](https://github.com/gnosis/prediction-market-agent/tree/main/prediction_market_agent/agents) and I recommend looking at the [DeployablePredictionProphetGPT4oAgent](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/agents/prophet_agent/deploy.py#L46C7-L46C44).

This is currently one of the best agents in the [leaderboard](https://presagio.pages.dev/leaderboard/agents), with 60% success rate and $834.73 in profits.

You can also play with this agent in [this Streamlit demo](https://pma-agent.ai.gnosisdev.com/?free_access_code=devcon), see what it's doing to get the final prediction.

For more details, there is also Dune Dashboard with detailed statistics. There you can filter for given time ranges, see daily stats, etc.

### Your task

1. Modify advanced agent in any way you deem best. You can change the LLM prompts, add more data sources, change the logic, etc.

You might not have enough of time to evaluate the agent throgouhly, so PoCs of good ideas with strong hypothesis of why they should work are very welcome as well.

If you want to use some 3rd party and it doesn't provide trial API keys for free, ping me and let's see if I can get them for you!

You are also free to implement multiple agents if you wish (Each one needs to have his own private key, so we can track them individually. Easiest is to have multiple copies of this repository, each with `.env` file of the given agent). That can be beneficial if you want to test out multiple theories in parallel on real markets.

### How to evaluate your agent

1. Every day, ~10 new markets are open and existing are resolved.

Set `bet_on_n_markets_per_run` and run your agent daily, it is the best evaluation.

By default, agent places only tiny bets, so no worries about spending too much!

1. Run the benchmark, this will get `n` markets from another prediction market platform, where mostly humans are trading. And generate markdown report against the human traders.

```bash
python scripts/simple_benchmark.py --n 10
```

(you have to add your agent into `agents` argument of `Benchmarker` class in the script)

1. Sometimes, the best thing is to manually observe what's going on -- print any outputs you can and observe what the agent is doing for some question.

For this, you can also use provided Streamlit demo, run it using `streamlit run src/app.py`.

### Will this be used in the end?

With your approval to use them, yes! If your agent achieves at least 50% accuracy on questions (proxy to know that he isn't losing money), I will add them to the production deployment with other agents and they will be live at Presagio.
