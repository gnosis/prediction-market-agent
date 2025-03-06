import json

from microchain import Function
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
        return f"""Use this function to get available jobs in a JSON dumped format.
You need to provide max bond value in {self.currency}, that is, how much you are willing to bond on the fact that you completed the job as required in the job description.
After completion, use SubmitJobResult.
"""

    @property
    def example_args(self) -> list[float]:
        return [1.0]

    def __call__(self, max_bond: float) -> str:
        jobs = self.market_type.job_class.get_jobs(limit=None)
        return json.dumps(
            [j.to_simple_job(max_bond=max_bond).model_dump() for j in jobs],
            indent=2,
            default=str,
        )


class SubmitJobResult(JobFunction):
    @property
    def description(self) -> str:
        return f"""Use this function to submit result of a job that you completed.
You need to submit this even if the job itself didn't ask for it explicitly, to prove that you completed the job.
"""

    @property
    def example_args(self) -> list[float | str]:
        return ["0x1", "GeneralAgent", 1.0, "I completed this job as described."]

    def __call__(
        self, job_id: str, agent_name: str, max_bond: float, result: str
    ) -> str:
        job = self.market_type.job_class.get_job(id=job_id)
        processed = job.submit_job_result(agent_name, max_bond, result)
        return json.dumps(
            processed.model_dump(),
            indent=2,
            default=str,
        )


JOB_FUNCTIONS: list[type[JobFunction]] = [
    GetJobs,
    SubmitJobResult,
]
