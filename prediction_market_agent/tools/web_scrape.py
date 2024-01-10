import bs4
import requests

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = (
        "Write a summary of the following text for {objective}:\n"
        '"{text}\n'
        "SUMMARY:"
    )
    t = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=t,
        combine_prompt=t,
        verbose=False,
    )
    return summary_chain.run(input_documents=docs, objective=objective)


def web_scrape(objective: str, url: str):
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    if len(text) > 10000:
        text = _summary(objective, text)
    return text


web_scraping_schema = {
    "type": "function",
    "function": {
        "name": "web_scraping",
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The objective that defines the content to be scraped from the website.",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the website to be scraped.",
                },
            },
            "required": ["query"],
        },
        "description": "Web scrape a URL to retrieve information relevant to the objective.",
    },
}


class WebScrapingTool:
    def __init__(self):
        self.fn = web_scrape
        self.schema = web_scraping_schema
