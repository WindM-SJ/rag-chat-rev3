# Ollama + RAG 로컬 챗봇 만들기 🤖

초보자도 따라할 수 있도록 단계별로 설명할게요!

---

## 🗺️ 전체 구조 이해하기

```
[내 문서 (PDF/TXT)] → [텍스트 분할] → [벡터DB 저장]
                                              ↓
[질문 입력] → [관련 문서 검색] → [Ollama LLM] → [답변 출력]
```

**RAG란?** 내 문서를 AI가 읽고, 그 내용을 바탕으로 대답하는 방식이에요.

---

## 📦 Step 1. 필요한 것 설치하기

### 1-1. Python 설치
👉 https://www.python.org/downloads/ 에서 **3.11 버전** 다운로드
- 설치 시 **"Add Python to PATH"** ✅ 반드시 체크!

### 1-2. Ollama 설치
👉 https://ollama.com 에서 다운로드 후 설치

설치 후 **VS Code 터미널** (Ctrl + \`) 열고 모델 다운로드:
```bash
# 한국어도 잘 되는 가벼운 모델 (약 4GB)
ollama pull llama3.2

# 잘 받아졌는지 확인
ollama list
```

---

## 📁 Step 2. 프로젝트 폴더 만들기

VS Code에서 새 폴더 열기 → 아래 구조로 파일 생성:

```
my-rag-chatbot/
├── docs/          ← 여기에 내 PDF, TXT 파일 넣기
├── app.py         ← 메인 챗봇 코드
└── requirements.txt
```

---

## 📝 Step 3. requirements.txt 작성

```txt
langchain
langchain-community
langchain-ollama
chromadb
pypdf
streamlit
sentence-transformers
```

터미널에서 설치:
```bash
pip install -r requirements.txt
```

---

## 💻 Step 4. 메인 코드 작성 (app.py)

```python
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# ── 페이지 설정 ──────────────────────────────
st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🤖")
st.title("🤖 나만의 로컬 RAG 챗봇")

# ── 문서 로딩 & 벡터DB 생성 (최초 1회만) ──────
@st.cache_resource
def load_vectorstore():
    docs_path = "./docs"
    all_docs = []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            all_docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            all_docs.extend(loader.load())

    if not all_docs:
        return None

    # 텍스트를 작은 조각으로 나누기
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(all_docs)

    # 벡터 DB에 저장 (한국어 지원 임베딩 모델)
    embeddings = SentenceTransformerEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

# ── QA 체인 생성 ────────────────────────────
@st.cache_resource
def get_qa_chain():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None

    llm = ChatOllama(model="llama3.2", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ── 채팅 UI ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = get_qa_chain()

if qa_chain is None:
    st.warning("📂 docs 폴더에 PDF 또는 TXT 파일을 넣어주세요!")
else:
    st.success("✅ 문서 로딩 완료! 질문해보세요.")

    # 이전 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 질문 입력
    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # AI 답변
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                st.write(answer)

                # 참고 문서 표시
                with st.expander("📄 참고한 문서 보기"):
                    for doc in result["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
```

---

## 🚀 Step 5. 실행하기

```bash
# 1. docs 폴더에 PDF나 TXT 파일 넣기

# 2. 챗봇 실행
streamlit run app.py
```

브라우저에서 **http://localhost:8501** 자동으로 열립니다! 🎉

---

## 🔧 자주 발생하는 오류 해결

| 오류 | 해결 방법 |
|------|----------|
| `ollama` 연결 안 됨 | 터미널에서 `ollama serve` 실행 |
| `pip` 명령어 없음 | Python PATH 재확인 후 재설치 |
| 한글 깨짐 | TXT 파일을 UTF-8로 저장 |
| 느린 응답 | RAM 16GB 이상 권장, 모델을 `llama3.2:1b`로 변경 |

---

## 💡 업그레이드

- **더 좋은 한국어 모델**: `ollama pull EEVE-Korean-10.8B` 
- **문서 추가**: `docs/` 폴더에 파일만 넣으면 자동 인식
- **문서 파일 중 hwp 파일도 업로드 할 수 있게 함
- **UI 커스텀**: Streamlit 공식 문서 참고

막히는 부분 있으면 편하게 물어보세요! 😊