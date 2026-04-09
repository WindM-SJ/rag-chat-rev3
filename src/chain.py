from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL, TEMPERATURE, NUM_PREDICT, NUM_CTX

_PROMPT = ChatPromptTemplate.from_template(
    """다음 문서 내용을 참고하여 질문에 간결하게 답변해주세요.
반드시 한글(Hangul)로만 답변하고, 한자(漢字)나 한문은 절대 사용하지 마세요.
핵심만 요점 정리하여 간단명료하게 답변하세요.
문서에 없는 내용은 모른다고 답변하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]"""
)


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
    )


def create_stream_chain(llm: ChatOllama):
    """스트리밍 가능한 LCEL 체인 반환 (context + question → str 토큰 스트림)."""
    return _PROMPT | llm | StrOutputParser()
