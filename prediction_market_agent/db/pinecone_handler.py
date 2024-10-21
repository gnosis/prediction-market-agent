import base64
import sys
import typing as t
from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import Index, Pinecone
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from tqdm import tqdm

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    PineconeMetadata,
)
from prediction_market_agent.utils import APIKeys

INDEX_NAME = "omen-index-text-embeddings-3-large"
T = t.TypeVar("T")


class PineconeHandler:
    vectorstore: VectorStore
    pc: Pinecone
    index: Index

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self.keys = APIKeys()
        self.model = model
        self.embeddings = OpenAIEmbeddings(
            api_key=self.keys.openai_api_key_secretstr_v1,
            model=model,
        )
        self.build_pinecone()
        self.build_vectorstore()

    def build_pinecone(self) -> None:
        self.pc = Pinecone(api_key=self.keys.pinecone_api_key.get_secret_value())
        self.index = self.pc.Index(INDEX_NAME)

    def build_vectorstore(self) -> None:
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.keys.pinecone_api_key.get_secret_value(),
            embedding=self.embeddings,
            index_name=INDEX_NAME,
        )

    def encode_text(self, text: str) -> str:
        """Encodes string using base64 and returns it as string"""
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    def decode_id(self, id: str) -> str:
        return base64.b64decode(id).decode("utf-8")

    def filter_markets_already_in_index(
        self, markets: list[OmenMarket]
    ) -> list[OmenMarket]:
        """
        This function filters out markets based on the market_title attribute of each market.
        It derives the ID of each market by encoding the market_title using base64 and
        then checks for the existence of these IDs in the index.

        The function then returns a list of markets that are not present in the index.

        """
        ids_market_map = {self.encode_text(m.question_title): m for m in markets}
        all_ids = list(ids_market_map.keys())
        ids_in_vec_db = self.get_existing_ids_in_index()
        missing_ids = set(all_ids).difference(ids_in_vec_db)
        filtered_markets = [ids_market_map[id] for id in missing_ids]
        return filtered_markets

    def get_existing_ids_in_index(self) -> list[str]:
        # index.list() returns [[id1,id2,...],[id4,id5,...]], hence the flattening.
        ids_in_vec_db = [y for x in self.index.list() for y in x]
        return ids_in_vec_db

    def insert_texts(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: Optional[list[dict[str, t.Any]]] = None,
    ) -> None:
        self.vectorstore.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

    @staticmethod
    def chunks(array: list[T], n_elements: int) -> t.Generator[list[T], None, None]:
        """Yield successive n_elements-sized chunks from array."""
        for i in range(0, len(array), n_elements):
            yield array[i : i + n_elements]

    @staticmethod
    def deduplicate_markets(markets: list[OmenMarket]) -> list[OmenMarket]:
        unique_market_titles: dict[str, OmenMarket] = {}
        for market in markets:
            if (
                market.title not in unique_market_titles
                or unique_market_titles[market.title].collateralVolume
                < market.collateralVolume
            ):
                unique_market_titles[market.question_title] = market

        return list(unique_market_titles.values())

    def update_markets(self) -> None:
        """We use the agent's run to add embeddings of new markets that don't exist yet in the
        vector DB."""
        self.insert_open_omen_markets_if_not_exists()

    def insert_open_omen_markets_if_not_exists(
        self, start_timestamp: int | None = None
    ) -> None:
        subgraph_handler = OmenSubgraphHandler()
        markets = subgraph_handler.get_omen_binary_markets_simple(
            limit=sys.maxsize,
            filter_by=FilterBy.NONE,
            created_after=DatetimeUTC.to_datetime_utc(start_timestamp)
            if start_timestamp
            else None,
            sort_by=SortBy.NEWEST,
        )

        markets_without_duplicates = self.deduplicate_markets(markets)
        missing_markets = self.filter_markets_already_in_index(
            markets=markets_without_duplicates
        )

        texts = []
        metadatas = []
        for m in missing_markets:
            texts.append(m.question_title)
            metadatas.append(PineconeMetadata.from_omen_market(m).model_dump())

        if texts:
            n_elements = 100
            chunked_texts = list(self.chunks(texts, n_elements))
            chunked_metadatas = list(self.chunks(metadatas, n_elements))
            for text_chunk, metadata_chunk in tqdm(
                zip(chunked_texts, chunked_metadatas), total=len(chunked_texts)
            ):
                ids_chunk = [self.encode_text(text) for text in text_chunk]
                logger.debug(f"Inserting {len(text_chunk)} into the vector database.")
                self.insert_texts(
                    ids=ids_chunk, texts=text_chunk, metadatas=metadata_chunk
                )

    def find_nearest_questions_with_threshold(
        self, limit: int, text: str, threshold: float = 0.25
    ) -> list[PineconeMetadata]:
        # Note that pagination is not implemented in the Pinecone client.
        # Hence we set a large limit and hope we get enough results that satisfy the threshold.
        documents_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query=text, k=limit, score_threshold=threshold
        )

        logger.debug(f"Found {len(documents_and_scores)} relevant documents.")
        return [
            PineconeMetadata.model_validate(doc.metadata)
            for doc, score in documents_and_scores
        ]
