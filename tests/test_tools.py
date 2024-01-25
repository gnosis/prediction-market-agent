import pytest
from tests.utils import RUN_PAID_TESTS
from prediction_market_agent.tools.google_search import GoogleSearchTool
from prediction_market_agent.tools.web_scrape import _summary, web_scrape


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_search_and_scrape() -> None:
    objective = "In 2024, will Israel conduct a military strike against Iran?"
    search = GoogleSearchTool().fn(query=objective)
    assert (
        "https://www.cnn.com/middleeast/live-news/israel-hamas-war-gaza-news-01-20-24/index.html"
        in search
    )
