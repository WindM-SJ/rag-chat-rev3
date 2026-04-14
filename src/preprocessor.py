"""
문서 전처리 모듈: docs/ 폴더의 파일을 markitdown으로 Markdown 변환 후
pretreated_docs/ 폴더에 저장합니다.

PDF를 직접 임베딩하는 것보다 Markdown으로 변환하면 구조(헤딩, 목록, 표 등)가
보존되어 청크 분할 품질과 검색 정확도가 향상됩니다.
"""

from pathlib import Path
from typing import Callable, Optional

from markitdown import MarkItDown

from src.config import DOCS_DIR, PRETREATED_DOCS_DIR

# markitdown이 지원하는 입력 확장자
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm"}


def convert_docs_to_markdown(
    docs_dir: Path = DOCS_DIR,
    output_dir: Path = PRETREATED_DOCS_DIR,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[Path]:
    """docs/ 폴더의 문서를 Markdown으로 변환하여 pretreated_docs/ 에 저장.

    Args:
        docs_dir: 원본 문서 폴더 경로.
        output_dir: 변환된 MD 파일을 저장할 폴더 경로.
        progress_callback: (완료 수, 전체 수, 현재 파일명) 를 받는 콜백.

    Returns:
        변환 성공한 MD 파일 경로 리스트.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 기존 MD 파일 전체 삭제 → 항상 최신 docs/ 상태로 재변환
    for existing_md in output_dir.glob("*.md"):
        existing_md.unlink()

    # 지원 확장자 파일만 수집
    source_files = sorted(
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_files:
        return []

    md_converter = MarkItDown()
    converted: list[Path] = []
    total = len(source_files)
    failed: list[str] = []

    for i, filepath in enumerate(source_files):
        if progress_callback:
            progress_callback(i, total, filepath.name)

        try:
            result = md_converter.convert(str(filepath))
            markdown_text = result.markdown or ""

            # 변환 결과가 비어있으면 스킵
            if not markdown_text.strip():
                failed.append(f"{filepath.name} (빈 결과)")
                continue

            out_path = output_dir / (filepath.stem + ".md")
            out_path.write_text(markdown_text, encoding="utf-8")
            converted.append(out_path)

        except Exception as exc:
            failed.append(f"{filepath.name} ({exc})")

    if progress_callback:
        progress_callback(total, total, "완료")

    if failed:
        # 변환 실패 목록을 로그 파일로 남김
        log_path = output_dir / "_conversion_errors.log"
        log_path.write_text("\n".join(failed), encoding="utf-8")

    return converted
