import os
from pathlib import Path

# HF 토큰: 캐시 파일에서 자동 로드 (huggingface-cli login 또는 직접 저장)
_hf_token_file = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
if os.path.exists(_hf_token_file):
    with open(_hf_token_file) as _f:
        os.environ.setdefault("HF_TOKEN", _f.read().strip())

import time
import streamlit as st

from src.loader import load_and_split
from src.vectorstore import create_vectorstore, get_retriever
from src.chain import get_llm, create_stream_chain

def _make_source_labels(docs: list) -> list[str]:
    """검색된 Document 리스트에서 '파일명 (p.N)' 형태의 레이블을 중복 없이 반환."""
    seen: set = set()
    labels: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = Path(source).name if source else "알 수 없는 파일"
        page = doc.metadata.get("page")
        label = f"{filename} (p.{page + 1})" if page is not None else filename
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(page_title="로컬 RAG 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 나만의 로컬 RAG 챗봇")
st.caption("docs/ 폴더의 문서를 기반으로 Gemma 2 (HuggingFace) LLM이 답변합니다.")

# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    if st.button("🔄 문서 재로드", use_container_width=True):
        # session_state에서 컴포넌트 제거 → 재초기화 트리거
        for key in ("retriever", "chain", "init_done"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.markdown("**docs/ 폴더에 PDF 또는 TXT 파일을 넣고 재로드하세요.**")
    st.markdown("""
    **지원 형식**
    - `.pdf`
    - `.txt` (UTF-8)

    **사용 모델**
    - LLM: `gemma-2-2b-it` (HuggingFace)
    - 임베딩: `ko-sroberta-multitask`
    """)

    if "llm_device" in st.session_state:
        st.divider()
        label = st.session_state.llm_device
        icon = "🟢" if "GPU" in label else "🟡"
        st.caption(f"{icon} LLM 실행 장치: `{label}`")

# ── 컴포넌트 초기화 (진행바 포함) ──────────────────────────────
def _run_init() -> tuple:
    """단계별 진행바를 표시하며 retriever, chain, device_label을 초기화."""

    init_placeholder = st.empty()

    with init_placeholder.container():
        st.info("⏳ 초기화를 시작합니다...")

        # 1단계 ─ 문서 로드 & 청크 분할
        with st.status("📂 문서 로딩 중...", expanded=True) as status:
            st.write("docs/ 폴더의 PDF·TXT 파일을 읽고 있습니다...")
            splits = load_and_split()

            if not splits:
                status.update(label="⚠️ 로딩할 문서가 없습니다.", state="error")
                return None, None, None

            total_chunks = len(splits)
            st.write(f"✅ 총 **{total_chunks}개** 청크로 분할 완료")

            # 2단계 ─ 임베딩
            st.write("🔢 임베딩 모델을 로드하고 벡터를 생성합니다...")
            progress_bar = st.progress(0.0, text="임베딩 준비 중...")

            def on_embed_progress(done: int, total: int) -> None:
                pct = done / total
                progress_bar.progress(
                    pct,
                    text=f"임베딩 중... {done}/{total} 청크 ({int(pct * 100)}%)",
                )

            vectorstore = create_vectorstore(splits, progress_callback=on_embed_progress)
            progress_bar.progress(1.0, text=f"✅ 임베딩 완료! ({total_chunks}개 청크)")
            retriever = get_retriever(vectorstore)

            # 3단계 ─ LLM 로드
            st.write("🤖 LLM(Gemma 2) 모델을 로드합니다... (최초 1회만 소요)")
            llm = get_llm()
            chain = create_stream_chain(llm)
            st.write(f"✅ LLM 로드 완료 — 실행 장치: `{llm.device_label}`")

            status.update(label="✅ 모든 초기화가 완료되었습니다!", state="complete")

    # 완료 후 플레이스홀더 제거
    init_placeholder.empty()
    return retriever, chain, llm.device_label


if "init_done" not in st.session_state:
    retriever, chain, device_label = _run_init()
    if retriever is None or chain is None:
        st.warning("📂 `docs/` 폴더에 PDF 또는 TXT 파일을 넣고 사이드바에서 **문서 재로드**를 눌러주세요.")
        st.stop()
    st.session_state.retriever = retriever
    st.session_state.chain = chain
    st.session_state.llm_device = device_label
    st.session_state.init_done = True
    st.rerun()  # 진행바 UI를 지우고 채팅 화면으로 전환

retriever = st.session_state.retriever
chain = st.session_state.chain

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
                    st.caption(src)

# ── 사용자 입력 ─────────────────────────────────────────────────
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # 1) 문서 검색
        with st.spinner("📂 관련 문서 검색 중..."):
            docs = retriever.invoke(prompt)
        context = "\n\n".join(doc.page_content for doc in docs)

        # 2) st.write_stream으로 스트리밍
        start = time.time()
        full_response = st.write_stream(
            chain.stream({"context": context, "question": prompt})
        )
        elapsed = round(time.time() - start, 1)
        st.caption(f"⏱️ {elapsed}초 소요")

        # 3) 참고 문서 표시 (파일명 + 페이지)
        source_labels = _make_source_labels(docs)
        if source_labels:
            with st.expander("📄 참고한 문서"):
                for label in source_labels:
                    st.caption(label)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_labels,
    })
