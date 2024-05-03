import requests
import tenacity
from bs4 import BeautifulSoup
from markdownify import markdownify
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.cache import persistent_inmemory_cache
from requests import Response


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True
)
@persistent_inmemory_cache
def fetch_html(url: str, timeout: int) -> Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    return response


def web_scrape(url: str, timeout: int = 10) -> str:
    """
    Taken from polywrap/predictionprophet

    https://github.com/polywrap/predictionprophet/blob/97aeea8f87e9b42da242d00d93ed5754bd64f21e/prediction_prophet/functions/web_scrape.py
    """
    try:
        response = fetch_html(url=url, timeout=timeout)

        if "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.content, "html.parser")

            [x.extract() for x in soup.findAll("script")]
            [x.extract() for x in soup.findAll("style")]
            [x.extract() for x in soup.findAll("noscript")]
            [x.extract() for x in soup.findAll("link")]
            [x.extract() for x in soup.findAll("head")]
            [x.extract() for x in soup.findAll("image")]
            [x.extract() for x in soup.findAll("img")]

            text: str = soup.get_text()
            text = markdownify(text)
            text = "  ".join([x.strip() for x in text.split("\n")])
            text = " ".join([x.strip() for x in text.split("  ")])

            return text
        else:
            logger.warning("Non-HTML content received")
            return ""

    except requests.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        return ""
