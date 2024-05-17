import base64
import sys
import typing as t
from datetime import datetime
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import Pinecone
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from tqdm import tqdm

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    PineconeMetadata,
)
from prediction_market_agent.utils import APIKeys

INDEX_NAME = "omen-markets"
T = t.TypeVar("T")


class PineconeHandler:
    def __init__(self) -> None:
        keys = APIKeys()
        self.pc = Pinecone(api_key=keys.pinecone_api_key.get_secret_value())
        self.index = self.pc.Index(INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(
            api_key=keys.openai_api_key.get_secret_value()
        )
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=keys.pinecone_api_key.get_secret_value(),
            embedding=self.embeddings,
            index_name=INDEX_NAME,
        )

    def encode_text(self, text: str) -> str:
        """Encodes string using base64 and returns it as string"""
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    def decode_id(self, id: str) -> str:
        return base64.b64decode(id).decode("utf-8")

    def find_texts_not_in_vec_db(self, texts: list[str]) -> dict[str, str]:
        ids_from_texts = [self.encode_text(text) for text in texts]
        # index.list() returns [[id1,id2,...],[id4,id5,...]], hence the flattening.
        ids_in_vec_db = [y for x in self.index.list() for y in x]
        missing_ids = set(ids_from_texts).difference(ids_in_vec_db)
        return {id: self.decode_id(id) for id in missing_ids}

    def insert_texts_if_not_exists(
        self, texts: list[str], metadatas: Optional[list[dict[str, t.Any]]] = None
    ) -> None:
        ids_to_texts = self.find_texts_not_in_vec_db(texts)
        ids, missing_texts = ids_to_texts.keys(), ids_to_texts.values()
        self.vectorstore.add_texts(
            texts=missing_texts, ids=list(ids), metadatas=metadatas
        )

    @staticmethod
    def chunks(array: list[T], n_elements: int) -> t.Generator[list[T], None, None]:
        """Yield successive n_elements-sized chunks from array."""
        for i in range(0, len(array), n_elements):
            yield array[i : i + n_elements]

    def insert_all_omen_markets_if_not_exists(
        self, created_after: datetime | None = None
    ) -> None:
        subgraph_handler = OmenSubgraphHandler()
        markets = subgraph_handler.get_omen_binary_markets_simple(
            limit=sys.maxsize,
            filter_by=FilterBy.NONE,
            sort_by=SortBy.NEWEST,
            created_after=created_after,
        )
        texts = [m.question_title for m in markets]
        metadatas = [
            PineconeMetadata.from_omen_market(market).model_dump() for market in markets
        ]
        if texts:
            n_elements = 100
            chunked_texts = list(self.chunks(texts, n_elements))
            chunked_metadatas = list(self.chunks(metadatas, n_elements))
            for text_chunk, metadata_chunk in tqdm(
                zip(chunked_texts, chunked_metadatas), total=len(chunked_texts)
            ):
                logger.debug(f"Inserting {len(text_chunk)} into the vector database.")
                self.insert_texts_if_not_exists(
                    texts=text_chunk, metadatas=metadata_chunk
                )

    def find_nearest_questions_with_threshold(
        self, limit: int, text: str, threshold: float = 0.7
    ) -> list[PineconeMetadata]:
        # Note that pagination is not implemented in the Pinecone client.
        # Hence we set a large limit and hope we get enough results that satisfy the threshold.
        documents_and_scores = self.vectorstore.similarity_search_with_score(
            query=text, k=int(limit * 5)
        )
        all_documents: list[Document] = []
        for doc, score in documents_and_scores:
            if len(all_documents) >= limit:
                break
            if score >= threshold:
                all_documents.append(doc)

        logger.debug(f"Found {len(all_documents)} relevant documents.")
        return [PineconeMetadata.model_validate(doc.metadata) for doc in all_documents]
