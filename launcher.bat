@echo off
setlocal enabledelayedexpansion

:: Music Module Launcher
:: Usage: launcher.bat [command] [options]

set "SCRIPT_DIR=%~dp0"

:: Determine Python executable (embedded first, then system)
set "PYTHON_EXE=%SCRIPT_DIR%python_embedded\python.exe"
if not exist "%PYTHON_EXE%" (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "delims=" %%i in ('where python') do set "PYTHON_EXE=%%i"
    )
)

:: Add portable Git to PATH if available
if exist "%SCRIPT_DIR%git_portable\cmd\git.exe" (
    set "PATH=%SCRIPT_DIR%git_portable\cmd;%PATH%"
)

:: Add FFmpeg to PATH if available (check various locations)
for /d %%D in ("%SCRIPT_DIR%ffmpeg\ffmpeg-*") do (
    if exist "%%D\bin\ffmpeg.exe" (
        set "PATH=%%D\bin;%PATH%"
    )
)
if exist "%SCRIPT_DIR%ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%SCRIPT_DIR%ffmpeg\bin;%PATH%"
)

:: Set working directory to script location (required for module imports)
cd /d "%SCRIPT_DIR%"

echo.
echo ========================================
echo   Music Module Launcher
echo ========================================
echo.

:: Check for python
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found.
    echo.
    echo   Run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

:: Verify core dependencies are installed
"%PYTHON_EXE%" -c "import fastapi, requests" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Required packages not installed.
    echo.
    echo   Run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

:: Parse command
set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=gui"

if /i "%COMMAND%"=="help" goto :show_help
if /i "%COMMAND%"=="setup" goto :run_setup
if /i "%COMMAND%"=="gui" goto :run_gui
if /i "%COMMAND%"=="api" goto :run_api
if /i "%COMMAND%"=="server" goto :run_api

echo Unknown command: %COMMAND%
goto :show_help

:show_help
echo.
echo Usage: launcher.bat [command] [options]
echo.
echo Commands:
echo   gui             Launch the Music Manager GUI (default)
echo   api, server     Start the Music API server directly
echo   setup           Run install.bat for full environment setup
echo   help            Show this help
echo.
echo API Server options (for 'api' command):
echo   --port PORT     Server port (default: 9150)
echo   --host HOST     Server host (default: 127.0.0.1)
echo.
echo Examples:
echo   launcher.bat                     Launch GUI
echo   launcher.bat api                 Start API server
echo   launcher.bat api --port 9200     Start API on port 9200
echo.
pause
exit /b 0

:run_setup
echo Running full environment setup...
call "%SCRIPT_DIR%install.bat"
pause
exit /b 0

:run_gui
echo Launching Music Manager GUI...
echo.
echo Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%SCRIPT_DIR%music_manager.py"

if errorlevel 1 (
    echo.
    echo ERROR: The Music Manager exited with an error.
    echo.
    pause
)
exit /b 0

:run_api
echo Starting Music API Server...

:: Collect remaining arguments
set "ARGS="
:collect_api_args
shift
if "%~1"=="" goto :start_api
set "ARGS=%ARGS% %~1"
goto :collect_api_args

:start_api
"%PYTHON_EXE%" "%SCRIPT_DIR%music_api_server.py" %ARGS%
if errorlevel 1 (
    echo.
    echo ERROR: API server exited with an error.
    echo.
    pause
)
endlocal
exit /b 0
