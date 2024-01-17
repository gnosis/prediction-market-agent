import requests
from bs4 import BeautifulSoup, Comment, Tag


def web_scrape_structured(url: str, remove_a_links: bool = True) -> str:
    """
    In contrast to `web_scrape`, which returns a block of text, maybe summarized with GPT,
    this functions returns as this (spaced for the context):

    ```
    A Historical look at Gnosis, GNO’s price
        GNO/USD Pair
            GNO
            USD
            16 January 2021
            106.76
            GNO
            USD
            16 January 2022
            398.11
            372.89
            GNO
            USD
            16 January 2023
    ```
    """
    if not url.startswith("http"):
        url = f"https://{url}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    page_content_html = response.text
    page_content_body_text_clean = pretty_html_from_page_content(
        page_content_html, remove_a_links=remove_a_links
    )
    return page_content_body_text_clean


def clean_soup(soup: Tag, remove_a_links: bool) -> Tag:
    # Remove all attributes from all tags (e.g. src, class, href, etc.), except for links
    for tag in soup.findAll(lambda x: len(x.attrs) > 0):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k == "href"}

    # Remove all hidden elements
    tags_to_remove = ["noscript", "script", "style"]
    if remove_a_links:
        tags_to_remove.append("a")
    for element_name in tags_to_remove:
        for element in soup.select(element_name):
            element.extract()

    # Remove all comments
    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()

    # Remove all elements with an empty content
    for element in soup.find_all():
        if len(element.get_text(strip=True)) == 0:
            element.extract()

    return soup


def prettify_html(html: str) -> str:
    return "\n".join(
        line
        for line in html.splitlines()
        if line.strip()
        and (
            not line.strip().startswith("<")
            or line.strip().startswith("<a")
            or line.strip() == "</a>"
        )
    )


def pretty_html_from_page_content(page_content_html: str, remove_a_links: bool) -> str:
    page_content_parsed = BeautifulSoup(page_content_html, "html.parser")
    page_content_body_text = clean_soup(
        page_content_parsed.find("body"), remove_a_links=remove_a_links
    ).prettify()
    page_content_body_text_clean = prettify_html(page_content_body_text)
    return page_content_body_text_clean


if __name__ == "__main__":
    print(
        web_scrape_structured(
            "https://ambcrypto.com/predictions/gnosis-price-prediction"
        )
    )
