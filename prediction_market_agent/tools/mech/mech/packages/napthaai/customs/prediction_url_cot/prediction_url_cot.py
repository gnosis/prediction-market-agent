from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from docstring_parser import parse
from googleapiclient.discovery import build
from itertools import islice
import json
import re
from io import BytesIO
import PyPDF2
from openai import OpenAI
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument
import requests
from requests.exceptions import RequestException, TooManyRedirects
from markdownify import markdownify as md
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from tiktoken import encoding_for_model

client: Optional[OpenAI] = None

class OpenAIClientManager:
    """Client context manager for OpenAI."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global client
        if client is not None:
            client.close()
            client = None

DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 300,
    "temperature": 0,
}
MAX_TOKENS = {
    "gpt-3.5-turbo-0125": 16385,
    "gpt-4-0125-preview": 8192,
}
ALLOWED_TOOLS = [
    "prediction-url-cot",
]
TOOL_TO_ENGINE = {tool: "gpt-4-0125-preview" for tool in ALLOWED_TOOLS}
DEFAULT_NUM_URLS = defaultdict(lambda: 3)
DEFAULT_NUM_QUERIES = defaultdict(lambda: 3)
NUM_URLS_PER_QUERY = 5
SPLITTER_CHUNK_SIZE = 1800*2
SPLITTER_OVERLAP = 50*4
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 1000
EMBEDDING_SIZE = 3072
SPLITTER_MAX_TOKENS = 1800
SPLITTER_OVERLAP = 50
NUM_NEIGHBORS = 4
HTTP_TIMEOUT = 20
HTTP_MAX_REDIRECTS = 5
HTTP_MAX_RETIES = 2
MAX_DOC_TOKENS = 10000


class OpenAISchema(BaseModel):  # type: ignore[misc]
    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.
        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion: Dict[str, Any]) -> "OpenAISchema":
        """
        Convert the response from OpenAI into the class instance
        Args:
            completion (dict): The response from OpenAI
        Returns:
            OpenAISchema: The instance of the class
        """

        message = completion.choices[0].message

        return cls.model_validate_json(
            message.function_call.arguments,
        )

class Document(BaseModel):
    text: str
    url: str
    embedding: Optional[List[float]] = None


class Queries(OpenAISchema):
    queries: List[str]

class Results(OpenAISchema):
    p_yes: float =  Field(description="Estimated probability that the event in the USER_QUESTION occurs.")
    p_no: float = Field(description="Estimated probability that the event in the USER_QUESTION does not occur.")
    confidence: float = Field(description="A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest confidence value; 1 maximum confidence value.")
    info_utility: float = Field(description="Utility of the information provided in ADDITIONAL_INFORMATION to help you make the prediction. 0 indicates lowest utility; 1 maximum utility.")
    prediction: Optional[str] = Field(description="The predicted outcome of the event in the USER_QUESTION. Can be 'yes', 'no', or 'I don't know'.")


PREDICTION_PROMPT = """
You are an AI expert in predicting events.
You are given a question and a document.
Your task is to predict whether the event in the question occurs based on the information in the document.
Only use the information in the document to make your prediction.
If you are not confident in your prediction, you can say "I don't know" in the prediction.
Please think step by step before your response.

USER_QUESTION:
```
{user_question}
```

DOCUMENT:
```
{document}
```
"""

URL_QUERY_PROMPT = """
You are an AI language model assistant. 
Your task is to generate {num_queries} different queries to retrieve relevant documents from the web.
Your response will be used to fetch information from the web to help you make a prediction about the event in the USER_PROMPT.
Please think step by step before your response.

USER_PROMPT:
```
{user_prompt}
```
"""

SYSTEM_PROMPT = """You are a world class algorithm for generating structured output from a given input."""

def search_google(query: str, api_key: str, engine: str, num: int) -> List[str]:
    service = build("customsearch", "v1", developerKey=api_key)
    search = (
        service.cse()
        .list(
            q=query,
            cx=engine,
            num=num,
        )
        .execute()
    )
    return [result["link"] for result in search["items"]]


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


def get_urls_from_queries(
    queries: List[str], api_key: str, engine: str, num: int
) -> List[str]:
    """Get URLs from search engine queries"""
    results = []
    for query in queries:
        try:
            for url in search_google(
                query=query,
                api_key=api_key,
                engine=engine,
                num=num,
            ):
                results.append(url)
        except Exception:
            pass
    unique_results = list(set(results))
    return unique_results


def extract_question(prompt: str) -> str:
    pattern = r'\"(.*?)\"'
    try:
        question = re.findall(pattern, prompt)[0]
    except Exception as e:
        question = prompt

    return question


def extract_text(
    html: str,
    num_words: Optional[int] = None,
) -> str:
    """Extract text from a single HTML document"""
    text = ReadabilityDocument(html).summary()

    # use html2text to convert HTML to markdown
    text = md(text, heading_style="ATX")

    if text is None:
        return None

    if num_words:
        text = " ".join(text.split()[:num_words])
    else:
        text = " ".join(text.split())

    doc = Document(text=text, url="")
    return doc


def extract_text_from_pdf(url: str, num_words: Optional[int] = None) -> str:
    """Extract text from a PDF document at the given URL."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return ValueError("URL does not point to a PDF document")

        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        doc = Document(text=text[:num_words] if num_words else text, date="", url=url)
        print(f"Using PDF: {url}: {doc.text[:300]}...")
        return doc
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def process_in_batches(
    urls: List[str], 
    window: int = 5, 
    timeout: int = HTTP_TIMEOUT,
    max_redirects: int = HTTP_MAX_REDIRECTS,
    retries: int = HTTP_MAX_RETIES,
) -> Generator[None, None, List[Tuple[Optional[Future], str]]]:
    """Iter URLs in batches with improved error handling and retry mechanism."""
    with ThreadPoolExecutor() as executor, requests.Session() as session:
        session.max_redirects = max_redirects
        for i in range(0, len(urls), window):
            batch = urls[i : i + window]
            futures = []
            for url in batch:
                future = None
                attempt = 0
                while attempt < retries:
                    try:
                        future = executor.submit(session.get, url, timeout=timeout)
                        break  
                    except (TooManyRedirects, RequestException) as e:
                        print(f"Attempt {attempt + 1} failed for {url}: {e}")
                        attempt += 1
                        if attempt == retries:
                            print(f"Max retries reached for {url}. Moving to next URL.")
                futures.append((future, url))
            yield futures


def extract_texts(urls: List[str], num_words: Optional[int] = None) -> List[Document]:
    """Extract texts from URLs with improved error handling, excluding failed URLs."""
    extracted_texts = []
    for batch in process_in_batches(urls=urls):
        for future, url in batch:
            if future is None:
                continue
            try:
                result = future.result()
                if result.status_code == 200:
                    # Check if URL ends with .pdf or content starts with %PDF
                    if url.endswith('.pdf') or result.content[:4] == b'%PDF':
                        doc = extract_text_from_pdf(url, num_words=num_words)
                    else:
                        doc = extract_text(html=result.text, num_words=num_words)
                    doc.url = url
                    extracted_texts.append(doc)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
    return extracted_texts


def multi_queries(
    client: OpenAI,
    prompt: str,
    engine: str,
    num_queries: int,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    temperature: int = DEFAULT_OPENAI_SETTINGS["temperature"],
    max_tokens: int = DEFAULT_OPENAI_SETTINGS["max_tokens"],
) -> List[str]:
    """Generate multiple queries for fetching information from the web."""

    url_query_prompt = URL_QUERY_PROMPT.format(
        user_prompt=prompt, num_queries=num_queries
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url_query_prompt},
    ]

    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=150,
        stop=None,
        functions=[Queries.openai_schema],
    )
    queries = Queries.from_response(response)

    # append the user's question to the list of queries
    queries.queries.append(prompt)

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )
        return queries.queries, counter_callback
    return queries.queries, None


def fetch_additional_information(
    client: OpenAI,
    prompt: str,
    engine: str,
    google_api_key: Optional[str],
    google_engine_id: Optional[str],
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    source_links: Optional[List[str]] = None,
    num_words: Optional[int] = None,
    num_urls: Optional[int] = None,
    num_queries: Optional[int] = DEFAULT_NUM_QUERIES,
    temperature: int = DEFAULT_OPENAI_SETTINGS["temperature"],
    max_tokens: int = DEFAULT_OPENAI_SETTINGS["max_tokens"],
) -> Tuple:
    """Fetch additional information from the web."""

    # generate multiple queries for fetching information from the web
    queries, counter_callback = multi_queries(
        client=client,
        prompt=prompt,
        engine=engine,
        num_queries=num_queries,
        counter_callback=counter_callback,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f"Queries: {queries}")

    # get the top URLs for the queries
    if not source_links:
        urls = get_urls_from_queries(
            queries=queries,
            api_key=google_api_key,
            engine=google_engine_id,
            num=NUM_URLS_PER_QUERY,
        )
        print(f"URLs: {urls}")

        urls = list(set(urls))

        # Extract text and dates from the URLs
        docs = extract_texts(
            urls=urls,
        )
    else:
        docs = []
        for url, content in islice(source_links.items(), num_urls or len(source_links)):
            doc = extract_text(html=content, num_words=num_words)
            doc.url = url
            docs.append(doc)

    # Remove None values from the list
    docs = [doc for doc in docs if doc]

    # remove empty documents ""
    filtered_docs = [doc for doc in docs if hasattr(doc, 'text') and doc.text != ""]

    return filtered_docs, counter_callback


def adjust_doc_tokens(
    doc: Document, 
    max_tokens: int, 
    engine: str = "gpt-4-0125-preview"
) -> Document:
    """Adjust the number of tokens in the document."""
    if count_tokens(doc.text, engine) > max_tokens:
        doc.text = " ".join(doc.text.split()[:max_tokens])
    return doc


def get_answer_from_doc(
    client: OpenAI,
    prompt: str,
    engine: str,
    doc: Document,
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    max_tokens: int = DEFAULT_OPENAI_SETTINGS["max_tokens"],
    temperature: int = DEFAULT_OPENAI_SETTINGS["temperature"],
):
    """Get an answer from the document."""
    # length of the document
    print(f"Length of the document before: {count_tokens(doc.text, engine)}")
    # Make sure each doc is with the max tokens
    doc = adjust_doc_tokens(
        doc=doc,
        max_tokens=MAX_DOC_TOKENS,
        engine=engine,
    )
    print(f"Length of the document after: {count_tokens(doc.text, engine)}")

    #Get the answer from the document
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PREDICTION_PROMPT.format(user_question=prompt, document=doc.text)},
    ]

    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=150,
        stop=None,
        functions=[Results.openai_schema],
        function_call={'name': 'Results'}
    )

    if counter_callback:
        counter_callback(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=engine,
            token_counter=count_tokens,
        )

    return Results.from_response(response), counter_callback, PREDICTION_PROMPT.format(user_question=prompt, document=doc.text)    


def get_answer(
    client: OpenAI,
    prompt: str,
    engine: str,
    additional_information: List[Document],
    counter_callback: Optional[Callable[[int, int, str], None]] = None,
    max_tokens: int = DEFAULT_OPENAI_SETTINGS["max_tokens"],
    temperature: int = DEFAULT_OPENAI_SETTINGS["temperature"],
):
    """Get an answer from the document."""

    for doc in additional_information:
        answer, counter_callback, prediction_prompt = get_answer_from_doc(
            client=client,
            prompt=prompt,
            engine=engine,
            doc=doc,
            counter_callback=counter_callback,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if answer.prediction in ["yes", "no"]:
            return answer, counter_callback, prediction_prompt
        
    return Results(p_yes=0.5, p_no=0.5, confidence=0.5, info_utility=0.5, prediction="I don't know"), counter_callback, prediction_prompt

def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the task"""
    with OpenAIClientManager(kwargs["api_keys"]["openai"]):

        tool = kwargs["tool"]
        prompt = extract_question(kwargs["prompt"])
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        num_words = kwargs.get("num_words", None)
        num_urls = kwargs.get("num_urls", DEFAULT_NUM_URLS[tool])
        num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES[tool])
        counter_callback = kwargs.get("counter_callback", None)
        api_keys = kwargs.get("api_keys", {})
        google_api_key = api_keys.get("google_api_key", None)
        google_engine_id = api_keys.get("google_engine_id", None)
        
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} is not supported.")

        engine = kwargs.get("model", TOOL_TO_ENGINE[tool])
        print(f"ENGINE: {engine}")
        # fetch additional information from the web
        additional_information, counter_callback = fetch_additional_information(
            client=client,
            prompt=prompt,
            engine=engine,
            google_api_key=google_api_key,
            google_engine_id=google_engine_id,
            counter_callback=counter_callback,
            source_links=kwargs.get("source_links", None),
            num_words=num_words,
            num_urls=num_urls,
            num_queries=num_queries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # get answer from the doc
        results, counter_callback, prediction_prompt = get_answer(
            client=client,
            prompt=prompt,
            engine=engine,
            additional_information=additional_information,
            counter_callback=counter_callback,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # convert the results to a dictionary
        pairs = str(results).split()
        result_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
            if key != "prediction":
                result_dict[key] = float(value)

        results = result_dict
        results = json.dumps(results)
        return results, prediction_prompt, None, counter_callback
