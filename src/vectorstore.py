from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_HUB, RETRIEVER_K


def get_embeddings():
    """한국어 임베딩 모델 반환 (로컬 경로 우선, HF Hub 폴백)."""
    model_name = (
        str(EMBEDDING_MODEL_PATH)
        if EMBEDDING_MODEL_PATH.exists()
        else EMBEDDING_MODEL_HUB
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def create_vectorstore(splits: list) -> Chroma:
    """Document 청크 리스트로 ChromaDB 벡터 스토어 생성."""
    embeddings = get_embeddings()
    return Chroma.from_documents(splits, embeddings)


def get_retriever(vectorstore: Chroma):
    """벡터 스토어에서 retriever 생성 (상위 k개 문서 반환)."""
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
