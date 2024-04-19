# Microchain agent behaviour diary

## GPT4

- Makes many reasoning steps, with coherent and good reasoning w.r.t betting strategy
- Almost always gets function calls correct
- Seems keen on betting large amounts, even when instructed not to!
- Seems keen to `Stop` program after max a couple bets. Doesn't use some of the functions (selling, getting existing positions)

## Local model setup

- In another terminal, run `ollama serve` to start the server e.g. `OLLAMA_HOST=127.0.0.1:11435 ollama serve`
- Pass the localhost address to the `api_base` arg, i.e. `api_base="http://localhost:11435/v1"`
- Pass the local model name (from `ollama list`) to the `model` arg.

## mixtral:8x7b-instruct-v0.1-q3_K_S

- Promising! Outputs some coherent reasoning. Chains several function calls together before starting to lose its way.
- Made several bad function calls, but followed up with reasoning to fix the function call, then made correct one.
- Didn't have a good view of all functions available. e.g.
  - tried to use `EndChat()` instead of `Stop()`
  - iterated through all markets to get market.p_yes, but didn't try to call a mech to predict its own p_yes.
- Questionable reasoning w.r.t betting strategy. Stated that a market with a large |p_yes - 0.5| was more likely to be mis-priced.

## llama3:latest

- Rubbish!
- Often replied with an empty string, aborting the program
- Couldn't recover after an incorrect function call
- 

## nous-hermes2:latest

- Took some system prompt massaging for it to get going
- Was able to recover from some bad function calls
- Correctly called a mech to predict its own p_yes, but reasoning about what to do with result unraveled
- Always falls over before placing a bet
