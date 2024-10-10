prompt_template = """You are given 2 Prediction Market titles, market 1 and market 2. Your job is to output a single float number between -1 and 1, representing the correlation between the event outcomes of markets 1 and 2.
                    Correlation can be understood as the conditional probability that market 2 resolves to YES, given that market 1 resolved to YES.
                    Correlation should be a float number between -1 and 1. 

                    [MARKET 1]
                    {main_market_question}

                    [MARKET 2]
                    {related_market_question}

                    Follow the formatting instructions below for producing an output in the correct format.
                    {format_instructions}
                    """
