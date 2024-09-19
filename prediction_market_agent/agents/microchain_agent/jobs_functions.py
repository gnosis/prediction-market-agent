import json

from microchain import Function
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.jobs.jobs import get_jobs
from prediction_market_agent_tooling.markets.data_models import Currency
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.utils import APIKeys


class JobFunction(Function):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        self.keys = keys
        self.market_type = market_type
        super().__init__()

    @property
    def currency(self) -> Currency:
        return self.market_type.market_class.currency


class GetJobs(JobFunction):
    @property
    def description(self) -> str:
        return """Use this function to get available jobs in a JSON dumped format.
You need to provide max bond value in xDai, that is, how much you are willing to bond on the fact that you completed the job as required in the job description.
"""

    @property
    def example_args(self) -> list[int]:
        return [10]

    def __call__(self, max_bond: xDai) -> str:
        jobs = get_jobs(self.market_type, limit=None)
        return json.dumps(
            [j.to_simple_job(max_bond=max_bond).model_dump() for j in jobs],
            indent=2,
            default=str,
        )


JOB_FUNCTIONS: list[type[Function]] = [
    GetJobs,
]
