import pytest
from prediction_market_agent import utils
from prediction_market_agent.tools.google_search import GoogleSearchTool
from prediction_market_agent.tools.web_scrape import _summary, web_scrape


@pytest.mark.skipif(utils.get_serp_api_key() is None, reason="No Serp API key")
def test_search_and_scrape() -> None:
    objective = "In 2024, will Israel conduct a military strike against Iran?"
    search = GoogleSearchTool().fn(query=objective)
    scraping = web_scrape(objective, search[0])
    print(_summary(objective, scraping))
