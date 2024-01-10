class AbstractAgent:
    def __init__(self):
        pass

    def run(self, objective: str) -> bool:
        """
        Execute the agent, and return the final result as a string.
        """
        raise NotImplementedError
