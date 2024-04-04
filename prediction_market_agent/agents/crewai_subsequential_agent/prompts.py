CREATE_OUTCOMES_FROM_SCENARIO_PROMPT = """
        You are asked about a SCENARIO, which can have multiple, mutually-exclusive outcomes.
        You should break down the SCENARIO into a list of possible outcomes.
        
        Example 1:
        - SCENARIO: Will Gnosis Pay reach 100,000 users by 2025?
        - Answer: '''
            - Gnosis Pay reaches less than 25,000 users by 2025
            - Gnosis Pay reaches more than 25,000 users but less than 50,000 users by 2025
            - Gnosis Pay reaches more than 50,000 users but less than 75,000 users by 2025
            - Gnosis Pay reaches 100,000 users or more by 2025
        '''

        Example 2:
        - SCENARIO: Will the price of Bitcoin go up again in the next 2 months?
        - Answer: '''
            - The price of crypto will go up in the next 2 months
            - The price of crypto will not go up in the next 2 months
        '''
        
        [SCENARIO]
        {scenario}
        """

CREATE_OUTCOMES_FROM_SCENARIO_OUTPUT = '''
  A list containing multiple bullet points. Each bullet point should start with '-'.
  Each bullet point should contain a possible outcome resulting from the
  provided SCENARIO. The produced outcomes should be mutually exclusive, i.e. only one of them should be true whereas
  the remaining ones should be false.
  '''

PROBABILITY_FOR_ONE_OUTCOME_PROMPT = """
                            Your task is to determine the probability of a prediction market affirmation being answered 'Yes' or 'No'.
        Use the sentence provided in 'SENTENCE' and follow these guidelines:
        * Focus on the affirmation inside double quotes in 'SENTENCE'.
        * The question must have only 'Yes' or 'No' outcomes. If not, respond with "Error".
        * Use the tools provided to aid your estimation.
        * Evaluate recent information more heavily than older information.
        * Your response should include:
            - "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
            - "p_yes": Probability that the sentence outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
            - "p_no": Probability that the sentence outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
            - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). 
            Confidence can be calculated based on the quality and quantity of data used for the estimation.
            
            Ensure p_yes + p_no equals 1.
        
            SENTENCE: {sentence}
"""

RESEARCH_OUTCOME_PROMPT = """
                Research and report on the following sentence:
                {sentence}
                Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.
                """
RESEARCH_OUTCOME_OUTPUT = """
            Return your answer in raw JSON format, with no special formatting such as newlines, as follows:
                {{"report": <REPORT>}}
                where <REPORT> is a free text field that contains a well though out justification 
                for the predicition based on the summary of your findings.
            """

FINAL_DECISION_PROMPT = """
Your task is to determine the probability of a given outcome being true.
Below you can find the , a list of tuples where the first element
is an outcome

[Outcomes with probabilities]
"""

FINAL_DECISION_PROMPT = """
Your task is to determine the probability of a binary outcome being answered 'Yes' or 'No'.
OUTCOMES_WITH_PROBABILITIES is provided, which contains a set of {number_of_outcomes} outcomes.
The object OUTCOMES_WITH_PROBABILITIES contains a list of tuple, where each tuple contains, as its
first element, an outcome, and as its second entry, a probability dictionary, having properties such
 as p_yes (probability that the outcome is true), p_no (probability that the outcome is false),
confidence (confidence level in the values of p_yes and p_no).
Observe that the outcomes contained inside OUTCOMES_WITH_PROBABILITIES are mutually exclusive, i.e.
only one of them can be true. 
You should determine the probability of the outcome OUTCOME_TO_ASSESS being true, 
considering the probabilities of the other related outcomes.

OUTCOME_TO_ASSESS: {outcome_to_assess}
OUTCOMES_WITH_PROBABILITIES: {outcomes_with_probabilities}
"""
PROBABILITY_CLASS_OUTPUT = """
        Your response should be a JSON string containing the following keys:
    - "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
    - "p_yes": Probability that the sentence outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
    - "p_no": Probability that the sentence outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
    - "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 
    1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.

  Ensure p_yes + p_no equals 1.
    
    Format your response in JSON format, including the keys "decision", "p_yes", "p_no" and "confidence".
    Only output the JSON-formatted string, nothing else.
"""
