import time
import streamlit as st

from src.loader import load_and_split
from src.vectorstore import create_vectorstore, get_retriever
from src.chain import get_llm, create_stream_chain

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(page_title="로컬 RAG 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 나만의 로컬 RAG 챗봇")
st.caption("docs/ 폴더의 문서를 기반으로 Qwen2.5 (HuggingFace) LLM이 답변합니다.")

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
    - LLM: `Qwen2.5-1.5B-Instruct` (HuggingFace)
    - 임베딩: `ko-sroberta-multitask`
    """)

# ── 컴포넌트 초기화 (캐싱) ──────────────────────────────────────
@st.cache_resource(show_spinner="모델 및 문서를 로딩 중... (최초 실행 시 수분 소요)")
def init_components():
    splits = load_and_split()
    if not splits:
        return None, None
    vectorstore = create_vectorstore(splits)
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    chain = create_stream_chain(llm)
    return retriever, chain


retriever, chain = init_components()

# ── 문서 없을 때 경고 ───────────────────────────────────────────
if retriever is None:
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
        # 1) 생각 중 표시 + 문서 검색
        status = st.empty()
        status.markdown("💭 **생각 중...**")

        docs = retriever.invoke(prompt)
        context = "\n\n".join(doc.page_content for doc in docs)

        # 예상 답변 시간 추정 (컨텍스트 길이 기반)
        ctx_chars = len(context)
        estimated_sec = max(5, min(60, ctx_chars // 60))
        status.info(f"💭 생각 중... · ⏱️ 예상 답변 시간: 약 {estimated_sec}초")

        # 2) 스트리밍 (첫 토큰 도착 시 상태 메시지 제거)
        response_box = st.empty()
        full_response = ""
        first_chunk = True
        start = time.time()

        for chunk in chain.stream({"context": context, "question": prompt}):
            if first_chunk:
                status.empty()
                first_chunk = False
            full_response += chunk
            response_box.markdown(full_response + "▌")

        response_box.markdown(full_response)
        elapsed = round(time.time() - start, 1)
        st.caption(f"⏱️ {elapsed}초 소요")

        # 3) 참고 문서 표시
        if docs:
            with st.expander("📄 참고한 문서"):
                for doc in docs:
                    st.caption(
                        doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": [doc.page_content for doc in docs],
    })
