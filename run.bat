@echo off
echo Starting FramePack-Studio...
echo.
echo VRAM Modes (auto-detected):
echo   low_vram   - ^< 26 GB free: DynamicSwap, transformer swaps per chunk
echo   medium_vram - 26-60 GB free: transformer stays on GPU, no per-chunk reloads (RTX 5090 32GB)
echo   high_vram  - ^> 60 GB free: all models on GPU (A100/H100)
echo.
echo To force high-vram mode (60GB+ cards): add --high-vram to the command below
echo.

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Cannot run studio.py.
    goto end
)

if exist "%cd%/.venv/Scripts/python.exe" (

"%cd%/.venv/Scripts/python.exe" studio.py --inbrowser

) else (

echo Error: Virtual Environment for Python not found. Did you install correctly?
goto end 

)

:end