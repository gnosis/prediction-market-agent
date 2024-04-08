# General agent

Idea of this agent is to be able to reason freely about which actions it should take in order to partake in Omen's markets. This includes actions such as `getMarkets`, `getYesTokens`, `getNoTokens`, `removeFunding`, etc.

This can be accomplished by integrating Omen's functions, available on [PMAT](https://github.com/gnosis/prediction-market-agent-tooling), into a CrewAI agent. Finally, such agent will adhere to PMA's agent framework, which basically means that a general method `run` is called every N hours, and inside that method the agent's reasoning should occur. 