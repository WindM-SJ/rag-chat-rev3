from __future__ import annotations

import threading
from typing import Any, Iterator, List, Optional

import torch
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.config import (
    LLM_MODEL_PATH,
    LLM_MODEL_HUB,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)

# 사용자 메시지 포맷 (context + question 부분만)
# 시스템 프롬프트는 LocalGemmaLLM 내부에서 chat_template에 주입됨
_PROMPT = PromptTemplate.from_template(
    """[참고 문서]
{context}

[질문]
{question}

[답변 지침]
반드시 아래 구조로 답변하세요.
1. 핵심 결론 (2~3문장)
2. 근거 및 상세 설명 (문서 내용을 바탕으로 구체적으로)
3. 절차·주의사항 또는 추가 정보 (해당하는 경우)"""
)

_SYSTEM_MSG = (
    "당신은 RAG 기반 문서 질의응답 도우미입니다.\n"
    "반드시 한글로만 답변하고, 꼭 필요한 전문 용어 외에 한자나 외국어는 사용하지 마세요.\n"
    "반드시 제공된 참고 문서(context)를 우선적으로 참고하여 답변하세요.\n\n"
    "답변 규칙:\n"
    "- 답변을 지나치게 짧게 끝내지 말 것\n"
    "- 먼저 핵심 결론을 말한 뒤, 그 이유와 근거를 자세히 설명할 것\n"
    "- 가능하면 절차 순서, 예시, 주의사항을 함께 설명할 것\n"
    "- 답변은 최소 4문장 이상으로 작성할 것\n"
    "- context에 없는 내용은 추측하지 말고, 확인 가능한 범위만 설명할 것\n"
    "- context가 부족하면 부족하다고 명시하고, 관련된 내용을 최대한 정리할 것"
)


class LocalGemmaLLM(LLM):
    """Hugging Face 인스트럭션 모델을 LangChain LLM으로 래핑."""

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _device_label: str = PrivateAttr(default="cpu")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_model()

    @property
    def device_label(self) -> str:
        """사이드바 등 UI에서 표시할 실행 장치 문자열."""
        return self._device_label

    def _load_model(self) -> None:
        """모델 로딩 전략 (device_map 분할 절대 금지):
        - CC >= 8.0 (RTX 30xx+) : 4-bit 양자화 (bitsandbytes)
        - CC 6.x~7.x (GTX 10xx/20xx): float16 전체 GPU
          → VRAM 부족 시 CPU 전체로 폴백 (split 없음)
        - CPU fallback : float32

        device_map="auto" 는 VRAM 부족 시 레이어를 CPU로 분할한다.
        GPU↔CPU PCIe 전송이 병목이 되어 GPU 단독/CPU 단독보다 느려지므로
        {"": "cuda:0"} 로 강제하고 OOM 발생 시 CPU로 완전 폴백한다.
        """
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}

        model_source = (
            str(LLM_MODEL_PATH) if LLM_MODEL_PATH.exists() else LLM_MODEL_HUB
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_source, **token_kwargs)

        if torch.cuda.is_available():
            cc_major = torch.cuda.get_device_capability()[0]
            gpu_name = torch.cuda.get_device_name(0)

            if cc_major >= 8:
                # Ampere 이상: 4-bit 양자화
                try:
                    from transformers import BitsAndBytesConfig
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        quantization_config=bnb,
                        device_map={"": "cuda:0"},
                        **token_kwargs,
                    )
                    self._device_label = f"GPU ({gpu_name}) · 4-bit"
                    print(f"[LLM] {self._device_label}")
                    return
                except Exception as e:
                    print(f"[LLM] 4-bit load failed: {e}")

            # Pascal/Turing (CC 6.x~7.x): float16 전체 GPU 강제
            # VRAM 여유 확보 후 로드, OOM 시 CPU 폴백
            torch.cuda.empty_cache()
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    device_map={"": "cuda:0"},   # split 없이 전체 GPU
                    **token_kwargs,
                )
                self._device_label = f"GPU ({gpu_name}) · float16"
                print(f"[LLM] {self._device_label}")
                return
            except Exception as e:
                print(f"[LLM] GPU float16 load failed ({e}), falling back to CPU")
                torch.cuda.empty_cache()

        # CPU fallback
        self._model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
            **token_kwargs,
        )
        self._device_label = "CPU · float32"
        print(f"[LLM] {self._device_label}")

    def _apply_chat_template(self, user_content: str) -> str:
        """모델의 공식 chat template 적용 (extra Q&A 생성 방지).
        system role 미지원 모델(Gemma 2 등)은 user 메시지에 시스템 지시를 병합.
        """
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user", "content": user_content},
            ]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Gemma 2 등 system role 미지원 모델: system 지시를 user에 병합
            messages = [
                {"role": "user", "content": f"{_SYSTEM_MSG}\n\n{user_content}"},
            ]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @property
    def _llm_type(self) -> str:
        return "local_instruction_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        formatted = self._apply_chat_template(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        # 생성된 토큰만 디코딩 (프롬프트 제외)
        generated = outputs[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        formatted = self._apply_chat_template(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            timeout=60.0,  # 생성 스레드 hang 방지
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(
            target=self._model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        # 토큰을 버퍼에 모아 단어·문장 경계에서 한 번에 yield.
        # 토큰 하나씩 yield하면 Streamlit이 매번 WebSocket 업데이트를 보내
        # 실제 생성 속도보다 렌더링이 훨씬 느리게 느껴진다.
        # 버퍼 크기나 플러시 문자 조건 중 하나가 충족되면 즉시 방출한다.
        _FLUSH_CHARS = frozenset(" \n\t.!?,。、，。：；!?()[]「」『』")
        _FLUSH_SIZE = 12  # 최소 누적 글자 수

        buffer = ""
        for token in streamer:
            buffer += token
            if len(buffer) >= _FLUSH_SIZE or (buffer and buffer[-1] in _FLUSH_CHARS):
                yield GenerationChunk(text=buffer)
                buffer = ""
        if buffer:
            yield GenerationChunk(text=buffer)

        thread.join()


def get_llm() -> LocalGemmaLLM:
    return LocalGemmaLLM()


def create_stream_chain(llm: LocalGemmaLLM):
    """스트리밍 가능한 LCEL 체인 반환 (context + question → str 토큰 스트림)."""
    return _PROMPT | llm | StrOutputParser()
