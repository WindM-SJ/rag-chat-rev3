import streamlit as st

from src.loader import load_and_split
from src.vectorstore import create_vectorstore, get_retriever
from src.chain import create_qa_chain

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(page_title="로컬 RAG 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 나만의 로컬 RAG 챗봇")
st.caption("docs/ 폴더의 문서를 기반으로 Ollama LLM이 답변합니다.")

# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    if st.button("🔄 문서 재로드", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**docs/ 폴더에 PDF 또는 TXT 파일을 넣고 재로드하세요.**")
    st.markdown("""
    **지원 형식**
    - `.pdf`
    - `.txt` (UTF-8)

    **사용 모델**
    - LLM: `llama3.2` (Ollama)
    - 임베딩: `ko-sroberta-multitask`
    """)

# ── 벡터 스토어 & QA 체인 초기화 (캐싱) ──────────────────────────
@st.cache_resource(show_spinner="문서를 벡터 DB에 로딩 중...")
def init_qa_chain():
    splits = load_and_split()
    if not splits:
        return None
    vectorstore = create_vectorstore(splits)
    retriever = get_retriever(vectorstore)
    return create_qa_chain(retriever)


qa_chain = init_qa_chain()

# ── 문서 없을 때 경고 ───────────────────────────────────────────
if qa_chain is None:
    st.warning("📂 `docs/` 폴더에 PDF 또는 TXT 파일을 넣고 사이드바에서 **문서 재로드**를 눌러주세요.")
    st.stop()

st.success("✅ 문서 로딩 완료! 질문해보세요.")

# ── 대화 히스토리 초기화 ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 이전 메시지 렌더링 ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 참고한 문서"):
                for src in msg["sources"]:
                    st.caption(src[:300] + "..." if len(src) > 300 else src)

# ── 사용자 입력 ─────────────────────────────────────────────────
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            sources = [doc.page_content for doc in result.get("source_documents", [])]

        st.write(answer)

        if sources:
            with st.expander("📄 참고한 문서"):
                for src in sources:
                    st.caption(src[:300] + "..." if len(src) > 300 else src)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
