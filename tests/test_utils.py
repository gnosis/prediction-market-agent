from crewai import Task

from prediction_market_agent.utils import disable_crewai_telemetry


def test_disable_crewai_telemetry():
    disable_crewai_telemetry()
    t = Task(
        description="foo",
        expected_output="bar",
    )
    assert not t._telemetry.task_started(task=t)
