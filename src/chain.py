from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from src.config import LLM_MODEL, TEMPERATURE


def create_qa_chain(retriever) -> RetrievalQA:
    """Ollama LLM과 retriever를 결합한 RetrievalQA 체인 반환."""
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
