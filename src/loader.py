from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.config import PRETREATED_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: Path = PRETREATED_DOCS_DIR) -> list:
    """pretreated_docs/ 폴더의 Markdown 파일을 로드하여 Document 리스트 반환.

    전처리된 MD 파일을 읽으며, 소스 메타데이터에 원본 파일명을 보존합니다.
    """
    all_docs = []

    if not docs_dir.exists():
        return all_docs

    for filepath in sorted(docs_dir.iterdir()):
        # 변환 오류 로그 파일은 스킵
        if filepath.name.startswith("_") or filepath.suffix.lower() != ".md":
            continue

        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            # source: md 파일 경로 그대로 유지 (표시용으로 stem을 사용)
            doc.metadata["source"] = str(filepath)
            # original_name: .md 확장자를 제거한 원본 파일명 (PDF 이름과 동일)
            doc.metadata["original_name"] = filepath.stem

        all_docs.extend(docs)

    return all_docs


def split_documents(docs: list) -> list:
    """Document 리스트를 Markdown 구조에 맞게 청크 단위로 분할.

    RecursiveCharacterTextSplitter.from_language(MARKDOWN) 를 사용하여
    헤딩·코드블록·단락 등 Markdown 구조를 우선 분리점으로 삼습니다.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_and_split(docs_dir: Path = PRETREATED_DOCS_DIR) -> list:
    """문서 로드 + 분할을 한 번에 수행."""
    docs = load_documents(docs_dir)
    if not docs:
        return []
    return split_documents(docs)
