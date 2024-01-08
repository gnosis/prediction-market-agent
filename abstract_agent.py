class AbstractAgent:
    def __init__(self):
        pass

    def run(self, objective: str) -> str:
        """
        Execute the agent, and return the final result as a string.
        """
        raise NotImplementedError
