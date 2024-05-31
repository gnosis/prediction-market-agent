import typing as t

from microchain import Function


class LearningFunction(Function):
    title: str
    knowledge: str

    def __init__(self) -> None:
        super().__init__()

    @property
    def description(self) -> str:
        return f"Use this function to learn about {self.title}"

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return self.knowledge


class LearnAboutLearning(LearningFunction):
    title = "learning"
    knowledge = """AI agents learn by using learning functions and then updating their prompt using the combination of first getting their prompt, then modyfing it and then updating it.
After each Learn function called, the agent should update its prompt to include the new knowledge.
The new knowledge needs to be about what was actually learned, not about just describing it.
Updating the prompt after learning session is absolute neccesity to make sure that the agent is able to use the new knowledge in the future.
Agent should keep a track record of what it has learned in the past, to not repeat the same learning session again and again.
"""


class LearnAboutWhatMakesAGoodPrompt(LearningFunction):
    title = "what makes a good prompt"
    knowledge = """A good prompt is a prompt that is clear, concise and informative. 
It should contain all the information that the agent needs to perform well.
Prompt such as "I know about X" isn't useful, the prompt should contain the actual knowledge about X, what it is, how it works, etc.
"""


class LearnAboutReasoning(LearningFunction):
    title = "reasoning"
    knowledge = """Reasoning is the process of thinking about something in a logical way in order to form a conclusion or judgment.
Reasoning function can be used for reasoning and usually it's helpful to call it quite often.
"""


class LearnAboutSurvival(LearningFunction):
    title = "survival"
    knowledge = """Survival is the state or fact of continuing to live or exist.
For an AI agent to survive, it must be able to continue to operate and perform its functions.
In order to operate, it must keep positive balance in its account.
Every action or function that the agent does is costing it some amount of money."""


class LearnAboutPredictionMarkets(LearningFunction):
    title = "prediction markets"
    knowledge = """By acting as an trader agent in prediction markets and using other available functions to operate on them,
an AI agent can create a profit to keep its balance positive."""


LEARNING_FUNCTIONS: list[t.Type[LearningFunction]] = [
    LearnAboutLearning,
    LearnAboutWhatMakesAGoodPrompt,
    LearnAboutReasoning,
    LearnAboutSurvival,
    LearnAboutPredictionMarkets,
]
