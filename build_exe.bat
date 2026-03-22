@echo off
REM ============================================
REM  MiniGPT-v2 EXE Builder
REM ============================================
echo.
echo ========================================
echo  MiniGPT-v2 EXE Builder
echo ========================================
echo.

REM Step 1: Check PyInstaller
echo [1/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)
echo Done.
echo.

REM Step 2: Build with PyInstaller
echo [2/4] Building EXE with PyInstaller (this may take 10-30 minutes)...
pyinstaller minigpt4.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed!
    echo Check the output above for errors.
    pause
    exit /b 1
)
echo Done.
echo.

REM Step 3: Create models directory and copy model files
echo [3/4] Setting up model files...
mkdir "dist\MiniGPT4\models" 2>nul
mkdir "dist\MiniGPT4\models\Llama-2-7b-chat-hf" 2>nul

if exist "Llama-2-7b-chat-hf" (
    echo Copying Llama-2-7b-chat-hf... (this will take a while, ~13GB)
    xcopy /E /Y "Llama-2-7b-chat-hf" "dist\MiniGPT4\models\Llama-2-7b-chat-hf\"
) else (
    echo WARNING: Llama-2-7b-chat-hf not found in current directory.
    echo You will need to manually copy it to dist\MiniGPT4\models\Llama-2-7b-chat-hf\
)

if exist "model\minigptv2_checkpoint.pth" (
    echo Copying minigptv2_checkpoint.pth...
    copy /Y "model\minigptv2_checkpoint.pth" "dist\MiniGPT4\models\"
) else (
    echo WARNING: model\minigptv2_checkpoint.pth not found.
    echo You will need to manually copy it to dist\MiniGPT4\models\
)

if exist "model\pretrained_minigpt4_llama2_7b.pth" (
    echo Copying pretrained_minigpt4_llama2_7b.pth...
    copy /Y "model\pretrained_minigpt4_llama2_7b.pth" "dist\MiniGPT4\models\"
)
echo Done.
echo.

REM Step 4: Copy example images (optional)
echo [4/4] Copying example images...
if exist "examples_v2" (
    xcopy /E /Y "examples_v2" "dist\MiniGPT4\examples_v2\"
)
echo Done.
echo.

echo ========================================
echo  BUILD COMPLETE!
echo ========================================
echo.
echo Output: dist\MiniGPT4\MiniGPT4.exe
echo.
echo To run: cd dist\MiniGPT4 ^& MiniGPT4.exe
echo.
echo NOTE: The target machine needs:
echo   - NVIDIA GPU with 8-12GB VRAM
echo   - NVIDIA CUDA drivers installed
echo   - 16GB+ RAM
echo.
echo NOTE: On first run, EVA-CLIP-G vision encoder weights
echo (~2GB) will be downloaded automatically.
echo.
pause
