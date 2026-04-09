from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: Path = DOCS_DIR) -> list:
    """docs/ 폴더의 PDF, TXT 파일을 로드하여 Document 리스트 반환."""
    all_docs = []

    if not docs_dir.exists():
        return all_docs

    for filepath in docs_dir.iterdir():
        if filepath.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(filepath))
            all_docs.extend(loader.load())
        elif filepath.suffix.lower() == ".txt":
            loader = TextLoader(str(filepath), encoding="utf-8")
            all_docs.extend(loader.load())

    return all_docs


def split_documents(docs: list) -> list:
    """Document 리스트를 청크 단위로 분할."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_and_split(docs_dir: Path = DOCS_DIR) -> list:
    """문서 로드 + 분할을 한 번에 수행."""
    docs = load_documents(docs_dir)
    if not docs:
        return []
    return split_documents(docs)
