# NFT Game Cheat Sheet

1. Start with the clean slate by running SQL commands from `clean_data.sql`
2. Add a new game round, for example

```python
python prediction_market_agent/agents/microchain_agent/nft_treasury_game/scripts/add_new_round.py "utcnow" "30" 
```

Will add a game round starting now and taking 30 minutes.

3. Prepare everything using the 

```python
python prediction_market_agent/agents/microchain_agent/nft_treasury_game/scripts/run_reset_game.py "https://remote-anvil-2.ai.gnosisdev.com"
```

This will distribute NFT keys, funds all agents with xDai, fund treasury with xDai and generate report if there was previous round.

This is also available as a cronjob in Kube, but it's more handy to run ti manually exactly when needed for workshops.

4. Start the agents by unsuspending them in the infrastructure repository.
