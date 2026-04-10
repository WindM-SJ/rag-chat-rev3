from pathlib import Path

# 문서 경로
DOCS_DIR = Path("./docs")

# 모델 디렉토리
MODEL_DIR = Path("./models")

# LLM 모델 - 로컬 경로 우선, 없으면 HF Hub 다운로드
# gemma-2-2b-it: Google Gemma 2 2B, HF 토큰 필요 (gated model)
LLM_MODEL_PATH = MODEL_DIR / "gemma-2-2b-it"
LLM_MODEL_HUB = "google/gemma-2-2b-it"

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
