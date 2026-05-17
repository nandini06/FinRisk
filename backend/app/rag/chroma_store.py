import os

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_chroma_db_path() -> str:
    return os.getenv("CHROMA_DB_PATH", "./chroma_db")


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=get_chroma_db_path())


def get_embedding_function(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_or_create_collection(
    name: str,
    embedding_function: EmbeddingFunction | None = None,
) -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)
