from pydantic import BaseModel


class Market(BaseModel):
    id: str
    question: str
    liquidity: float

    def __repr__(self):
        return f"Market: {self.question}"
