"""
RAG Chat Rev3 Launcher
더블클릭 시 Ollama + Streamlit 자동 시작 후 브라우저 오픈
"""
import subprocess
import sys
import os
import time
import webbrowser
import threading
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# ── 앱 디렉토리 (exe 기준 또는 스크립트 기준) ──────────────
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

PORT = 8501


# ── 유틸리티 ────────────────────────────────────────────────
def find_ollama() -> str | None:
    path = shutil.which("ollama")
    if path:
        return path
    default = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs" / "Ollama" / "ollama.exe"
    )
    return str(default) if default.exists() else None


def find_streamlit() -> str | None:
    # 1) 같은 폴더 venv
    venv_st = APP_DIR / ".venv" / "Scripts" / "streamlit.exe"
    if venv_st.exists():
        return str(venv_st)
    # 2) 시스템 PATH
    return shutil.which("streamlit")


# ── 런처 윈도우 ─────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Chat Rev3")
        self.geometry("420x220")
        self.resizable(False, False)
        self.configure(bg="#1E2630")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._proc: subprocess.Popen | None = None
        self._build_ui()
        threading.Thread(target=self._start, daemon=True).start()
        self.mainloop()

    # ── UI ──────────────────────────────────────────────────
    def _build_ui(self):
        # 헤더
        tk.Label(
            self, text="🤖  RAG Chat Rev3",
            bg="#1E2630", fg="white",
            font=("Segoe UI", 17, "bold"),
        ).pack(pady=(22, 4))

        # 상태 레이블
        self._status = tk.StringVar(value="시작 중...")
        tk.Label(
            self, textvariable=self._status,
            bg="#1E2630", fg="#95A5A6",
            font=("Segoe UI", 10),
        ).pack()

        # 진행 바
        self._bar = ttk.Progressbar(self, length=340, mode="indeterminate")
        self._bar.pack(pady=12)
        self._bar.start(12)

        # 버튼 행
        btn_row = tk.Frame(self, bg="#1E2630")
        btn_row.pack(pady=4)

        self._open_btn = tk.Button(
            btn_row, text="브라우저 열기",
            font=("Segoe UI", 10), state=tk.DISABLED,
            bg="#2ECC71", fg="white", activebackground="#27AE60",
            relief=tk.FLAT, bd=0, padx=18, pady=6,
            command=self._open_browser,
        )
        self._open_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_row, text="종료",
            font=("Segoe UI", 10),
            bg="#E74C3C", fg="white", activebackground="#C0392B",
            relief=tk.FLAT, bd=0, padx=18, pady=6,
            command=self._on_close,
        ).pack(side=tk.LEFT, padx=6)

    def _set_status(self, text: str, color: str = "#95A5A6"):
        self.after(0, lambda: self._status.set(text))
        self.after(0, lambda: self.nametowidget(
            self._status._tk).configure()  # tk 내부 색 변경은 별도 레이블로 처리
        )
        # 색상 변경은 직접 위젯 참조로
        for widget in self.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("textvariable") == str(self._status):
                self.after(0, lambda w=widget, c=color: w.configure(fg=c))

    def _ready(self):
        self._bar.stop()
        self._bar.configure(mode="determinate", value=100)
        self._status.set(f"✅  실행 중  →  http://localhost:{PORT}")
        self._open_btn.configure(state=tk.NORMAL)
        self._open_browser()

    def _error(self, msg: str):
        self._bar.stop()
        self.after(0, lambda: messagebox.showerror("오류", msg, parent=self))

    # ── 실행 로직 ────────────────────────────────────────────
    def _start(self):
        try:
            self._start_ollama()
            self._start_streamlit()
            self.after(0, self._ready)
        except Exception as exc:
            self.after(0, lambda: self._error(str(exc)))

    def _start_ollama(self):
        self._status.set("Ollama 서비스 확인 중...")
        ollama = find_ollama()
        if not ollama:
            return  # Ollama 없으면 건너뜀

        try:
            result = subprocess.run(
                [ollama, "ps"],
                capture_output=True, timeout=4,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                return  # 이미 실행 중
        except Exception:
            pass

        self._status.set("Ollama 시작 중...")
        subprocess.Popen(
            [ollama, "serve"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        time.sleep(3)

    def _start_streamlit(self):
        self._status.set("챗봇 서버 시작 중...")
        st = find_streamlit()
        if not st:
            raise RuntimeError(
                "Streamlit을 찾을 수 없습니다.\n\n"
                "아래 명령으로 패키지를 먼저 설치하세요:\n"
                "  pip install -r requirements.txt"
            )

        self._proc = subprocess.Popen(
            [
                st, "run", str(APP_DIR / "app.py"),
                "--server.port", str(PORT),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Streamlit 준비 대기 (최대 30초)
        for _ in range(30):
            time.sleep(1)
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read().decode(errors="ignore")
                raise RuntimeError(f"Streamlit 시작 실패:\n{stderr[:600]}")
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{PORT}", timeout=1)
                return  # 준비 완료
            except Exception:
                pass

    def _open_browser(self):
        webbrowser.open(f"http://localhost:{PORT}")

    # ── 종료 ────────────────────────────────────────────────
    def _on_close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self.destroy()


if __name__ == "__main__":
    Launcher()
