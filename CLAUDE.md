# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

Ollama + LangChain + ChromaDB + Streamlit을 사용한 **로컬 RAG 챗봇** 구현 프로젝트입니다. `claude chat.md`는 구현 참고용 가이드입니다.

## 프로젝트 설정 (가이드 기반)

가이드를 따라 실제 프로젝트를 만들 경우:

```bash
# 의존성 설치
pip install -r requirements.txt

# Ollama 모델 다운로드
ollama pull llama3.2

# 앱 실행
streamlit run app.py
```

## 기술 스택

- **LLM**: Ollama (`llama3.2`, 로컬 실행)
- **임베딩**: SentenceTransformer (`jhgan/ko-sroberta-multitask`, 한국어 특화)
- **벡터 DB**: ChromaDB (인메모리)
- **문서 로딩**: LangChain `PyPDFLoader`, `TextLoader`
- **UI**: Streamlit
- **체인**: LangChain `RetrievalQA`

## 아키텍처

```
docs/ (PDF/TXT) → RecursiveCharacterTextSplitter (chunk=500, overlap=50)
               → SentenceTransformerEmbeddings → ChromaDB

질문 → ChromaDB retriever (k=3) → ChatOllama → 답변 + 참고 문서
```

Streamlit의 `@st.cache_resource`로 벡터 DB와 QA 체인을 최초 1회만 초기화합니다.
