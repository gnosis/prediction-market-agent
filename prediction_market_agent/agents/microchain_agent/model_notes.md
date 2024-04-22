# Microchain Agent Model Behaviour Diary

## Proprietary models

### GPT4

- Makes many reasoning steps, with coherent and good reasoning w.r.t betting strategy
- Almost always gets function calls correct
- Seems keen on betting large amounts, even when instructed not to!
- Seems keen to `Stop` program after max a couple bets. Doesn't use some of the functions (selling, getting existing positions)

## Local models

### Setup

- Instructions are for Ollama, but you can use any library that allows you to set up a local OpenAI-compatible server.
- Download Ollama [here](https://ollama.com/download/mac)
- In another terminal, run `ollama serve` to start the server. You can set the address and port with the `OLLAMA_HOST` env var, e.g.:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve`
```

- Run the script, passing in the API address and model name as arguments. Note that you must have downloaded these model weights in advance via `ollama run <model_name`:

```bash
python prediction_market_agent/agents/microchain_agent/microchain_agent.py --api-base "http://localhost:11435/v1" --model "nous-hermes2:latest"
```

Note that the first call to the model will be slow, as the model weights are loaded from disk into GPU memory/RAM.

## [mixtral:8x7b-instruct-v0.1-q3_K_S](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-q3_K_S)

- Promising! Outputs some coherent reasoning. Chains several function calls together before starting to lose its way.
- Made several bad function calls, but followed up with reasoning to fix the function call, then made correct one.
- Didn't have a good view of all functions available. e.g.
  - tried to use `EndChat()` instead of `Stop()`
  - iterated through all markets to get market.p_yes, but didn't try to call a mech to predict its own p_yes.
- Questionable reasoning w.r.t betting strategy. Stated that a market with a large |p_yes - 0.5| was more likely to be mis-priced.

## [llama3:latest](https://ollama.com/library/llama3)

- Couldn't get any useful function calls from it.
- Often replied with an empty string, aborting the program
- Couldn't recover after an incorrect function call

## [nous-hermes2:latest](https://ollama.com/library/nous-hermes2)

- Took some system prompt massaging for it to get going
- Was able to recover from some bad function calls
- Correctly called a mech to predict its own p_yes, but reasoning about what to do with result unraveled
- Always falls over before placing a bet
