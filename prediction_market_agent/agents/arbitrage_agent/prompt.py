prompt_template = """Given two markets, MARKET 1 and MARKET 2, provide a boolean value that represents the correlation between these two markets' outcomes. Return True if the outcomes are perfectly or nearly perfectly correlated, meaning there is a high probability that both markets resolve to the same outcome. Return False if the correlation is weak or non-existent.
Correlation can also be understood as the conditional probability that market 2 resolves to YES, given that market 1 resolved to YES.
In addition to the boolean value, explain the reasoning behind your decision. 

[MARKET 1]
{main_market_question}

[MARKET 2]
{related_market_question}

Follow the formatting instructions below for producing an output in the correct format.
{format_instructions}"""
