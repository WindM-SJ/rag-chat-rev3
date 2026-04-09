from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.config import EMBEDDING_MODEL, RETRIEVER_K


def get_embeddings():
    """한국어 SentenceTransformer 임베딩 모델 반환."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore(splits: list) -> Chroma:
    """Document 청크 리스트로 ChromaDB 벡터 스토어 생성."""
    embeddings = get_embeddings()
    return Chroma.from_documents(splits, embeddings)


def get_retriever(vectorstore: Chroma):
    """벡터 스토어에서 retriever 생성 (상위 k개 문서 반환)."""
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
