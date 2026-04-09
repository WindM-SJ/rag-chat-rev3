from pathlib import Path

# 문서 경로
DOCS_DIR = Path("./docs")

# 임베딩 모델 (한국어 특화)
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# Ollama LLM 설정
LLM_MODEL = "llama3.2"
TEMPERATURE = 0

# 텍스트 분할 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 검색 설정
RETRIEVER_K = 3
