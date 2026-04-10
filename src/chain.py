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
{question}"""
)

_SYSTEM_MSG = (
    "당신은 문서 기반 질의응답 전문가입니다.\n"
    "반드시 한글(Hangul)로만 답변하고, 한자(漢字)나 한문은 절대 사용하지 마세요.\n"
    "제공된 문서 내용만 참고하여 핵심을 간결하게 답변하세요.\n"
    "문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 답변하세요."
)


class LocalGemmaLLM(LLM):
    """Hugging Face 인스트럭션 모델을 LangChain LLM으로 래핑."""

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        """모델 로딩 - GPU 사용 가능 시 4bit 양자화, 아니면 CPU."""
        model_source = (
            str(LLM_MODEL_PATH) if LLM_MODEL_PATH.exists() else LLM_MODEL_HUB
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_source)

        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    quantization_config=bnb,
                    device_map="auto",
                )
                return
            except Exception:
                pass  # fallback to CPU

        # CPU fallback
        self._model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
        )

    def _apply_chat_template(self, user_content: str) -> str:
        """모델의 공식 chat template 적용 (extra Q&A 생성 방지)."""
        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": user_content},
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

        for text in streamer:
            yield GenerationChunk(text=text)

        thread.join()


def get_llm() -> LocalGemmaLLM:
    return LocalGemmaLLM()


def create_stream_chain(llm: LocalGemmaLLM):
    """스트리밍 가능한 LCEL 체인 반환 (context + question → str 토큰 스트림)."""
    return _PROMPT | llm | StrOutputParser()
