@echo off
echo Starting FramePack-Studio...

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Cannot run studio.py.
    goto end
)

if exist "%cd%/venv_py312/Scripts/python.exe" (

"%cd%/venv_py312/Scripts/python.exe" studio.py --inbrowser

) else (

echo Error: Virtual Environment for Python not found. Did you install correctly?
goto end 

)

:end