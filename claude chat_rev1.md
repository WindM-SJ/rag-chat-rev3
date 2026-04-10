# Hugging Face 기반 RAG 챗봇 Plan (풀패키지 배포용)

## 🎯 변경 목표

| 기존 (Ollama) | 변경 (Hugging Face 직접) |
|--------------|------------------------|
| 사용자가 Ollama 별도 설치 필요 | Python 환경만 있으면 실행 |
| 모델이 Ollama 서버에 의존 | 모델 파일이 프로젝트에 포함 |
| 배포 시 설치 가이드 복잡 | 풀패키지 압축 → 압축해제 → 실행 |

---

## ⚖️ Llama 3.2 vs Gemma 2 비교 분석

| 항목 | Llama 3.2 (3B) | Gemma 2 (2B) |
|------|---------------|--------------|
| **라이선스** | Llama Community License | **Apache 2.0** ✅ |
| **상업적 사용** | MAU 7억 초과 시 Meta 허가 필요 | **완전 자유** ✅ |
| **모델 용량** | ~6GB | **~4GB** ✅ |
| **VRAM 요구** | 8GB+ | **6GB+** ✅ |
| **한국어 성능** | 양호 | 양호 |
| **배포 제약** | HF 접근 시 Meta 약관 동의 필요 | **없음** ✅ |
| **추론 속도** | 보통 | **빠름** ✅ |

### 🏆 최종 선택: **Gemma 2 (2B-it)**

**선택 이유:**
1. **Apache 2.0 라이선스** - 사내/외부 배포 시 법적 리스크 제로
2. **작은 용량** - 풀패키지 배포 시 다운로드/저장 부담 감소
3. **낮은 하드웨어 요구** - 더 많은 사용자 PC에서 실행 가능
4. **HF 다운로드 시 별도 약관 동의 불필요** - 자동화된 배포 스크립트 가능

---

## 📦 수정된 전체 구조

```
my-rag-chatbot/
├── models/                    ← Gemma 2 모델 파일 (배포 시 포함)
│   ├── gemma-2-2b-it/
│   └── ko-sroberta-multitask/ ← 임베딩 모델
├── docs/                      ← 사용자 문서
├── app.py                     ← 메인 챗봇 코드
├── download_models.py         ← 최초 모델 다운로드 스크립트
├── requirements.txt
└── run.bat / run.sh           ← 원클릭 실행 스크립트
```

---

## 📝 Step 1. requirements.txt (수정)

```txt
langchain
langchain-community
chromadb
pypdf
streamlit
sentence-transformers
transformers
torch
accelerate
bitsandbytes
```

---

## 📥 Step 2. 모델 다운로드 스크립트 (download_models.py)

```python
"""
최초 1회만 실행 - 모델을 로컬에 다운로드
배포 시에는 models/ 폴더째 포함하면 됨
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

MODEL_DIR = "./models"

def download_llm():
    """Gemma 2 모델 다운로드"""
    print("📥 Gemma 2 모델 다운로드 중... (약 4GB)")
    save_path = os.path.join(MODEL_DIR, "gemma-2-2b-it")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype="auto",
        device_map="auto"
    )
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"✅ 저장 완료: {save_path}")

def download_embedding():
    """임베딩 모델 다운로드"""
    print("📥 임베딩 모델 다운로드 중...")
    save_path = os.path.join(MODEL_DIR, "ko-sroberta-multitask")
    
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    model.save(save_path)
    print(f"✅ 저장 완료: {save_path}")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_llm()
    download_embedding()
    print("\n🎉 모든 모델 다운로드 완료!")
    print("이제 models/ 폴더를 프로젝트에 포함해서 배포하세요.")
```

---

## 💻 Step 3. 메인 코드 (app.py) - 수정본

```python
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Optional, List

# ── 경로 설정 ───────────────────────────────
MODEL_PATH = "./models/gemma-2-2b-it"
EMBEDDING_PATH = "./models/ko-sroberta-multitask"
DOCS_PATH = "./docs"

# ── 커스텀 LLM 클래스 (Hugging Face 모델 래핑) ──
class LocalGemmaLLM(LLM):
    model: object = None
    tokenizer: object = None
    
    def __init__(self):
        super().__init__()
        # 4bit 양자화로 메모리 절약
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            local_files_only=True  # 오프라인 실행 보장
        )
    
    @property
    def _llm_type(self) -> str:
        return "local_gemma"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 제거하고 응답만 반환
        return response[len(prompt):].strip()

# ── 페이지 설정 ──────────────────────────────
st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🤖")
st.title("🤖 나만의 로컬 RAG 챗봇")
st.caption("Gemma 2 기반 | 완전 오프라인 실행")

# ── 문서 로딩 & 벡터DB 생성 ──────────────────
@st.cache_resource
def load_vectorstore():
    all_docs = []
    
    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            all_docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            all_docs.extend(loader.load())
    
    if not all_docs:
        return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(all_docs)
    
    # 로컬 임베딩 모델 사용
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_PATH,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

@st.cache_resource
def get_qa_chain():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    
    llm = LocalGemmaLLM()
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

# 모델 로딩 상태
with st.spinner("🔄 모델 로딩 중... (최초 실행 시 1-2분 소요)"):
    qa_chain = get_qa_chain()

if qa_chain is None:
    st.warning("📂 docs 폴더에 PDF 또는 TXT 파일을 넣어주세요!")
else:
    st.success("✅ 준비 완료! 질문해보세요.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                st.write(answer)
                
                with st.expander("📄 참고한 문서 보기"):
                    for doc in result["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
```

---

## 🚀 Step 4. 실행 스크립트

**run.bat (Windows)**
```batch
@echo off
echo 🤖 RAG 챗봇 시작 중...
python -m streamlit run app.py
pause
```

**run.sh (Mac/Linux)**
```bash
#!/bin/bash
echo "🤖 RAG 챗봇 시작 중..."
python -m streamlit run app.py
```

---

## 📦 Step 5. 배포 패키지 구성

```
배포용_RAG챗봇.zip
├── models/           ← 약 5GB (Gemma 2 + 임베딩)
├── docs/             ← 빈 폴더 또는 샘플 문서
├── app.py
├── requirements.txt
├── run.bat
├── run.sh
└── 설치가이드.txt
```

**설치가이드.txt 내용:**
```
1. Python 3.11 설치 (python.org)
2. 터미널에서: pip install -r requirements.txt
3. docs/ 폴더에 PDF, TXT 파일 넣기
4. run.bat 더블클릭 (Windows) 또는 ./run.sh (Mac/Linux)
```

---

## ✅ 변경사항 요약

| 항목 | 기존 (Ollama) | 변경 (Hugging Face) |
|------|--------------|-------------------|
| 모델 소스 | ollama pull | transformers 직접 로딩 |
| 외부 의존성 | Ollama 서버 필수 | 없음 (Python만) |
| 오프라인 실행 | ollama serve 필요 | local_files_only=True |
| 배포 방식 | 설치 가이드 복잡 | 압축파일 배포 |
| 모델 | llama3.2 | gemma-2-2b-it |

---

이 Plan대로 진행할까요? 각 단계별 상세 구현이 필요하면 말씀해주세요!