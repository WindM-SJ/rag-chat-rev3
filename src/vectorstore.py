from typing import Callable, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_HUB, RETRIEVER_K

# 한 번에 임베딩할 청크 수 — 너무 크면 메모리 압박, 너무 작으면 오버헤드
EMBED_BATCH_SIZE = 32


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
        encode_kwargs={"batch_size": EMBED_BATCH_SIZE},
    )


def create_vectorstore(
    splits: list,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Chroma:
    """Document 청크 리스트로 ChromaDB 벡터 스토어를 배치 단위로 생성.

    Args:
        splits: 임베딩할 Document 청크 리스트.
        progress_callback: (완료 수, 전체 수) 를 받는 콜백. 각 배치 완료 후 호출.
    """
    embeddings = get_embeddings()
    total = len(splits)

    if total == 0:
        return Chroma.from_documents([], embeddings)

    # 첫 번째 배치로 Chroma 인스턴스 초기화
    first_batch = splits[:EMBED_BATCH_SIZE]
    vectorstore = Chroma.from_documents(first_batch, embeddings)

    done = len(first_batch)
    if progress_callback:
        progress_callback(done, total)

    # 나머지 배치를 순차 추가
    for start in range(EMBED_BATCH_SIZE, total, EMBED_BATCH_SIZE):
        batch = splits[start : start + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        done = min(start + EMBED_BATCH_SIZE, total)
        if progress_callback:
            progress_callback(done, total)

    return vectorstore


def get_retriever(vectorstore: Chroma):
    """벡터 스토어에서 retriever 생성 (상위 k개 문서 반환)."""
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
