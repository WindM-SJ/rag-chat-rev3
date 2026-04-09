@echo off
chcp 65001 >nul
echo ========================================
echo   RAG Chat Rev3 - EXE 빌드
echo ========================================
echo.

echo [1/3] PyInstaller 설치 확인...
python -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo 오류: pip 실행 실패. Python이 설치되어 있는지 확인하세요.
    pause & exit /b 1
)

echo [2/3] EXE 빌드 중...
pyinstaller --onefile ^
            --windowed ^
            --name "RAGChatRev3" ^
            --clean ^
            launcher.py

if errorlevel 1 (
    echo.
    echo 오류: 빌드 실패
    pause & exit /b 1
)

echo.
echo [3/3] 완료!
echo.
echo 생성된 파일: dist\RAGChatRev3.exe
echo.
echo 사용 방법:
echo   dist\RAGChatRev3.exe 를 app.py 와 같은 폴더에 복사한 후 실행
echo.
pause
