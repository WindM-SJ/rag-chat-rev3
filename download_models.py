"""
최초 1회만 실행 - 모델을 로컬에 다운로드
배포 시에는 models/ 폴더째 포함하면 됩니다.

사용법:
    python download_models.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

MODEL_DIR = "./models"


def download_llm():
    """Gemma 2 모델 다운로드 (약 4GB)"""
    print("📥 Gemma 2 2B-it 모델 다운로드 중... (약 4GB, 시간이 걸립니다)")
    save_path = os.path.join(MODEL_DIR, "gemma-2-2b-it")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype="auto",
        device_map="auto",
    )

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"✅ LLM 저장 완료: {save_path}")


def download_embedding():
    """한국어 임베딩 모델 다운로드"""
    print("📥 임베딩 모델 다운로드 중...")
    save_path = os.path.join(MODEL_DIR, "ko-sroberta-multitask")

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    model.save(save_path)
    print(f"✅ 임베딩 저장 완료: {save_path}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_embedding()
    download_llm()
    print("\n🎉 모든 모델 다운로드 완료!")
    print("이제 models/ 폴더를 프로젝트에 포함해서 배포하거나, 그냥 앱을 실행하세요.")
