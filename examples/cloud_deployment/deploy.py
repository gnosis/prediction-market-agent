import os

from prediction_market_agent.deploy.deploy import (
    deploy_to_gcp,
    remove_deployed_gcp_function,
    run_deployed_gcp_function,
    schedule_deployed_gcp_function,
)
from prediction_market_agent.deploy.utils import gcp_function_is_active
from prediction_market_agent.markets.all_markets import MarketType
from prediction_market_agent.utils import get_keys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = deploy_to_gcp(
        requirements_file=f"{current_dir}/../../pyproject.toml",
        extra_deps=[
            "git+https://github.com/gnosis/prediction-market-agent.git@evan/deploy-agent"
        ],
        function_file=f"{current_dir}/agent.py",
        market_type=MarketType.MANIFOLD,
        api_keys={"MANIFOLD_API_KEY": get_keys().manifold},
        memory=512,
    )

    # Check that the function is deployed
    assert gcp_function_is_active(fname)

    # Run the function
    response = run_deployed_gcp_function(fname)
    assert response.ok

    # Schedule the function
    schedule_deployed_gcp_function(fname, sleep_time=60 * 10)

    # Delete the function
    remove_deployed_gcp_function(fname)
