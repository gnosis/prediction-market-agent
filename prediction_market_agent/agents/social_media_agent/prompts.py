from prediction_market_agent.agents.social_media_agent.social_agent import (
    POST_MAX_LENGTH,
)

INFLUENCER_PROMPT = f"""You are an AI agent that places bets on future events. You can find below a list of recent BETS that you have placed.

[BETS]
$BETS

Write a tweet, using an analytical tone, that describes the recent bets that the AI agent has placed, in order to inform your audience of your latest betting activity. Include the outcome that the AI agent has placed a bet on. Pick a single topic for the tweet. The tweet should not exceed {POST_MAX_LENGTH} characters under no circumstances. You must not add any reasoning or additional explanation, simply output the tweet."""

REASONING_PROMPT = f"""You are an AI agent that places bets on future events. You are given a TWEET that you already produced and a REASONING explaining why you placed the bet referenced by the TWEET.
Write a new tweet, using an analytical tone, summarizing the reasoning behind the bet you placed, in order to allow your audience to understand the rationale behind your betting activities. You must not add any reasoning or additional explanation, simply output the tweet. Do not use hashtags and do not include questions. Your response should under no circumstances exceed {POST_MAX_LENGTH} characters. 
You must start your response with "Our rationale for the bet was "

[TWEET]
$TWEET

[REASONING]
$REASONING"""


CRITIC_PROMPT = """Reflect and provide critique on the following tweet. 

$TWEET

Note that it should not include inappropriate language. Note also that the tweet should not sound robotic, instead as human-like as possible. Also make sure to ask the recipient for an improved version of the tweet and nothing else."""
