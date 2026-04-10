from pathlib import Path

# 문서 경로
DOCS_DIR = Path("./docs")

# 모델 디렉토리
MODEL_DIR = Path("./models")

# LLM 모델 - 로컬 경로 우선, 없으면 HF Hub 다운로드
# Qwen2.5-1.5B-Instruct: Apache 2.0, 인증 불필요, 한국어 지원
# Gemma 2 사용 시: LLM_MODEL_PATH = MODEL_DIR / "gemma-2-2b-it"
#                  LLM_MODEL_HUB  = "google/gemma-2-2b-it"  (HF 토큰 필요)
LLM_MODEL_PATH = MODEL_DIR / "qwen2.5-1.5b-instruct"
LLM_MODEL_HUB = "Qwen/Qwen2.5-1.5B-Instruct"

# 임베딩 모델 (한국어 특화) - 로컬 경로 우선, 없으면 HF Hub 다운로드
EMBEDDING_MODEL_PATH = MODEL_DIR / "ko-sroberta-multitask"
EMBEDDING_MODEL_HUB = "jhgan/ko-sroberta-multitask"

# 텍스트 분할 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 검색 설정
RETRIEVER_K = 2

# LLM 생성 설정
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
