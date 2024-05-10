CREATE_REQUIRED_CONDITIONS_PROMPT = """
You are asked to create a list of conditions that must be met for a SCENARIO to be true.

For example, if the SCENARIO is "Will person X win the election?", the conditional SCENARIOs could be: "Person X must be a candidate in the election", "Person X must not be disqualified from the election", etc.

Write short, concise SCENARIOs, with at most 10 words each, in a form of a question.
Write up to {n_scenarios} other SCENARIOs that must be met for the SCENARIO to be true.

[SCENARIO]
{scenario}
"""

CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT = """
You are asked about a SCENARIO, which can have either 'YES' or 'NO' outcomes.

However, based on the SCENARIO, you should create a list of other possible SCENARIOS, these SCENARIOS should also have either 'YES' or 'NO' possible outcomes.

Here are a few examples of SCENARIOs and then SCENARIOs derived from them:

Example 1:
- SCENARIO: Will Gnosis Pay reach 100,000 users by 2025?
- Possible other SCENARIOs:
  - Will Gnosis Pay have less than 25,000 users by 2025?
  - Will Gnosis Pay reach more than 25,000 users but less than 50,000 users by 2025?
  - Will Gnosis Pay reach more than 50,000 users but less than 75,000 users by 2025?
  - Will Gnosis Pay reach 100,000 users or more by 2025?
  - Will Gnosis Pay reach 100,000 users or more by 2028?
  - Will Gnosis Pay reach 100,000 users or more by 2030?

Example 2:
- SCENARIO: Will the price of Bitcoin go up again in the next 2 months?
- Possible other SCENARIOs:
  - Will the price of crypto go up in the next 2 months?
  - Will the price of crypto will go up in the next 6 months?
  - The price of crypto will not go up in the next 2 months
  - The price of crypto will not go up in the next 1 months


Be creative and think of all other possible SCENARIOs derived from the given SCENARIO. 
Always include also the opposite of the SCENARIO, for example, if the SCENARIO is "Will the price of Bitcoin go up again in the next 2 months?", also include "The price of Bitcoin will not go up again in the next 2 months".
Don't be limited by the examples above. 
If the given SCENARIO is about some numerical value, think of different ranges or values that can be derived from it and don't be biased towards similar values, think of all possible reasonable values.
Write up to {n_scenarios} possible SCENARIOs derived from the given SCENARIO.

[SCENARIO]
{scenario}
"""

LIST_OF_SCENARIOS_OUTPUT = """
A list containing multiple bullet points. Each bullet point should start with '-'.
Each bullet point should contain a possible outcome resulting from the provided SCENARIO.
"""

PROBABILITY_FOR_ONE_OUTCOME_PROMPT = """
Your task is to determine the probability of a prediction market affirmation being answered 'Yes' or 'No'.
Use the sentence provided in 'SENTENCE' and follow these guidelines:
- Focus on the affirmation inside double quotes in 'SENTENCE'.
- The question must have only 'Yes' or 'No' outcomes. If not, respond with "Error".
- Use the tools provided to aid your estimation.
- Evaluate recent information more heavily than older information.

SENTENCE: {sentence}
"""

RESEARCH_OUTCOME_PROMPT = """
Research and report on the following sentence:
{sentence}
Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.
"""
RESEARCH_OUTCOME_WITH_PREVIOUS_OUTPUTS_PROMPT = """
Research and report on the following sentence:
{sentence}
Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.

You can use the following estimates, but you should not take them as the ground truth, they were generated independently and can be incorrect:
{previous_scenarios_with_probabilities}
"""
RESEARCH_OUTCOME_OUTPUT = """
Return your answer in raw JSON format, with no special formatting such as newlines, as follows:
{{"report": <REPORT>}}
where <REPORT> is a free text field that contains a well though out justification 
for the predicition based on the summary of your findings.
The report must not be longer than 1000 words.
"""

FINAL_DECISION_PROMPT = """
Your task is to determine the probability of a SCENARIO being answered 'Yes' or 'No'.
SCENARIOS_WITH_PROBABILITIES is provided, which contains a set of {number_of_scenarios} SCENARIOs.
The object SCENARIOS_WITH_PROBABILITIES contains a list of tuple, where each tuple contains, as its
first element, an scenario, and as its second entry, a probability dictionary, having properties such
as p_yes (probability that the scenario is true), p_no (probability that the scenario is false),
confidence (confidence level in the values of p_yes and p_no).
You should determine the probability of the SCENARIO SCENARIO_TO_ASSESS being true, 
considering the probabilities of the other related SCENARIOs.

SCENARIOS_WITH_PROBABILITIES: 
{scenarios_with_probabilities}

SCENARIO_TO_ASSESS: {scenario_to_assess}
"""

PROBABILITY_CLASS_OUTPUT = """
Your response should be a JSON string containing the following keys:
- "reasoning": A free text field that contains a well though out justification for the prediction.
- "decision": The decision you made. Either `y` (for `Yes`) or `n` (for `No`).
- "p_yes": Probability that the sentence outcome will be `Yes`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "p_no": Probability that the sentence outcome will be `No`. Ranging from 0 (lowest probability) to 1 (maximum probability).
- "confidence": Indicating the confidence in the estimated probabilities you provided ranging from 0 (lowest confidence) to 1 (maximum confidence). Confidence can be calculated based on the quality and quantity of data used for the estimation.

Do not surround the output object with escape quotes, line breaks nor '''. 
Do not output any other text, only the JSON object.

Ensure p_yes + p_no equals 1.
"""
