import base64
import hashlib

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import Pinecone
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

from prediction_market_agent.utils import APIKeys


INDEX_NAME = "omen-markets"


# ToDo - Move to PMAT
class PineconeHandler:
    def __init__(self):
        k = APIKeys()
        self.pc = Pinecone(api_key=k.pinecone_api_key.get_secret_value())
        self.index = self.pc.Index(INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(api_key=k.openai_api_key.get_secret_value())
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=k.pinecone_api_key.get_secret_value(),
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

    def insert_texts_if_not_exists(self, texts: list[str]):
        ids_to_texts = self.find_texts_not_in_vec_db(texts)
        ids, missing_texts = ids_to_texts.keys(), ids_to_texts.values()
        self.vectorstore.add_texts(texts=missing_texts, ids=list(ids))

    def find_nearest_questions(self, limit: int, text: str) -> list[str]:
        documents = self.vectorstore.similarity_search(query=text, k=limit)
        return [doc.page_content for doc in documents]
