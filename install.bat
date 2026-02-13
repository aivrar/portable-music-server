@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Music Module - Autonomous Installer
echo ============================================
echo.
echo   This script sets up everything from scratch:
echo   - Embedded Python 3.10 (no system Python needed)
echo   - Portable Git (no system Git needed)
echo   - Portable FFmpeg (for audio processing)
echo   - All Python dependencies
echo   - Then launches the Music Manager GUI
echo.

set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python_embedded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "GIT_DIR=%SCRIPT_DIR%git_portable"
set "GIT_EXE=%GIT_DIR%\cmd\git.exe"
set "FFMPEG_DIR=%SCRIPT_DIR%ffmpeg"
set "FFMPEG_EXE=%FFMPEG_DIR%\bin\ffmpeg.exe"

set "PYTHON_VERSION=3.10.11"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_ZIP=%SCRIPT_DIR%python_embedded.zip"

set "GIT_VERSION=2.47.1"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v%GIT_VERSION%.windows.1/MinGit-%GIT_VERSION%-64-bit.zip"
set "GIT_ZIP=%SCRIPT_DIR%git_portable.zip"

set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FFMPEG_URL_FALLBACK=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_ZIP=%SCRIPT_DIR%ffmpeg_portable.zip"

:: ============================================
:: Step 1: Download Embedded Python
:: ============================================
if exist "%PYTHON_EXE%" echo [OK] Embedded Python already installed. && goto :step_pth

echo [1/8] Downloading Python %PYTHON_VERSION% embedded...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"

if not exist "%PYTHON_ZIP%" goto :err_python

echo [1/8] Extracting Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"

if not exist "%PYTHON_EXE%" goto :err_python
del "%PYTHON_ZIP%" 2>nul

:: ============================================
:: Step 2: Configure ._pth for site-packages
:: ============================================
:step_pth
echo [2/8] Configuring Python for package installation...

if not exist "%PYTHON_DIR%\Lib\site-packages" mkdir "%PYTHON_DIR%\Lib\site-packages"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$pthFiles = Get-ChildItem '%PYTHON_DIR%\python*._pth'; if ($pthFiles.Count -gt 0) { $pth = $pthFiles[0]; $zipName = (Get-ChildItem '%PYTHON_DIR%\python*.zip' | Select-Object -First 1).Name; if (-not $zipName) { $zipName = 'python310.zip' }; $content = @($zipName, '.', 'Lib', 'Lib\site-packages', '', 'import site'); $content | Set-Content -Path $pth.FullName -Encoding ASCII; Write-Host '   Configured:' $pth.Name } else { Write-Host 'WARNING: No ._pth file found' }"

:: ============================================
:: Step 3: Bootstrap pip
:: ============================================
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% equ 0 echo [OK] pip already available. && goto :step_tkinter

echo [3/8] Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"

if not exist "%PYTHON_DIR%\get-pip.py" goto :err_pip

echo [3/8] Installing pip...
"%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py" >nul 2>&1
if errorlevel 1 goto :err_pip

del "%PYTHON_DIR%\get-pip.py" 2>nul
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1
echo [OK] pip installed.

:: ============================================
:: Step 4: Set up tkinter
:: ============================================
:step_tkinter
"%PYTHON_EXE%" -c "import _tkinter" >nul 2>&1
if %errorlevel% equ 0 echo [OK] tkinter already available. && goto :step_git

echo [4/8] Setting up tkinter for GUI...

set "TCLTK_MSI_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/amd64/tcltk.msi"
set "TCLTK_MSI=%SCRIPT_DIR%_tcltk.msi"
set "TCLTK_DIR=%SCRIPT_DIR%_tcltk_extract"

echo   Downloading tcltk.msi...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '!TCLTK_MSI_URL!' -OutFile '!TCLTK_MSI!'"

if not exist "!TCLTK_MSI!" echo WARNING: Failed to download tcltk.msi. GUI may not work. && goto :step_git

echo   Extracting tkinter components...
if exist "!TCLTK_DIR!" rmdir /S /Q "!TCLTK_DIR!" 2>nul
powershell -NoProfile -Command "Start-Process -FilePath 'msiexec.exe' -ArgumentList '/a','!TCLTK_MSI!','/qn','TARGETDIR=!TCLTK_DIR!' -Wait -NoNewWindow"

:: Copy DLLs
if exist "!TCLTK_DIR!\DLLs\_tkinter.pyd" (
    copy /Y "!TCLTK_DIR!\DLLs\_tkinter.pyd" "%PYTHON_DIR%\" >nul 2>&1
    copy /Y "!TCLTK_DIR!\DLLs\tcl86t.dll" "%PYTHON_DIR%\" >nul 2>&1
    copy /Y "!TCLTK_DIR!\DLLs\tk86t.dll" "%PYTHON_DIR%\" >nul 2>&1
    if exist "!TCLTK_DIR!\DLLs\zlib1.dll" copy /Y "!TCLTK_DIR!\DLLs\zlib1.dll" "%PYTHON_DIR%\" >nul 2>&1
)

:: Copy Lib/tkinter/
if exist "!TCLTK_DIR!\Lib\tkinter" (
    if exist "%PYTHON_DIR%\Lib\tkinter" rmdir /S /Q "%PYTHON_DIR%\Lib\tkinter" 2>nul
    xcopy /E /I /Y "!TCLTK_DIR!\Lib\tkinter" "%PYTHON_DIR%\Lib\tkinter" >nul 2>&1
)

:: Copy tcl/ library
if exist "!TCLTK_DIR!\tcl" (
    if exist "%PYTHON_DIR%\tcl" rmdir /S /Q "%PYTHON_DIR%\tcl" 2>nul
    xcopy /E /I /Y "!TCLTK_DIR!\tcl" "%PYTHON_DIR%\tcl" >nul 2>&1
)

:: Cleanup
rmdir /S /Q "!TCLTK_DIR!" 2>nul
del "!TCLTK_MSI!" 2>nul

:: Verify
"%PYTHON_EXE%" -c "import _tkinter" >nul 2>&1
if errorlevel 1 echo WARNING: tkinter setup failed. GUI may not work.
if not errorlevel 1 echo [OK] tkinter setup complete.

:: ============================================
:: Step 5: Download Portable Git
:: ============================================
:step_git
if exist "%GIT_EXE%" echo [OK] Portable Git already installed. && goto :step_ffmpeg

echo [5/8] Downloading portable Git %GIT_VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%GIT_ZIP%'"

if not exist "%GIT_ZIP%" echo WARNING: Failed to download Git. && goto :step_ffmpeg

echo [5/8] Extracting portable Git...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%GIT_ZIP%' -DestinationPath '%GIT_DIR%' -Force"
del "%GIT_ZIP%" 2>nul

:: ============================================
:: Step 6: Download Portable FFmpeg
:: ============================================
:step_ffmpeg
set "FFMPEG_FOUND=0"
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" set "FFMPEG_FOUND=1"
if exist "%FFMPEG_DIR%\ffmpeg.exe" set "FFMPEG_FOUND=1"
for /d %%D in ("%FFMPEG_DIR%\ffmpeg-*") do if exist "%%D\bin\ffmpeg.exe" set "FFMPEG_FOUND=1"

if "%FFMPEG_FOUND%"=="1" echo [OK] FFmpeg already installed. && goto :step_path

echo [6/8] Downloading portable FFmpeg...
echo   Trying primary source (gyan.dev)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%'"

if not exist "%FFMPEG_ZIP%" (
    echo   Primary source failed. Trying fallback (GitHub BtbN)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%FFMPEG_URL_FALLBACK%' -OutFile '%FFMPEG_ZIP%'"
)

if not exist "%FFMPEG_ZIP%" echo WARNING: Failed to download FFmpeg from both sources. Audio conversion may not work. && goto :step_path

echo [6/8] Extracting portable FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_DIR%' -Force"
del "%FFMPEG_ZIP%" 2>nul

:: ============================================
:: Step 7: Set up PATH and install requirements
:: ============================================
:step_path
if exist "%GIT_EXE%" set "PATH=%GIT_DIR%\cmd;%PATH%" && echo [OK] Portable Git added to PATH.

for /d %%D in ("%FFMPEG_DIR%\ffmpeg-*") do (
    if exist "%%D\bin\ffmpeg.exe" set "PATH=%%D\bin;%PATH%" && echo [OK] FFmpeg added to PATH.
)
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" set "PATH=%FFMPEG_DIR%\bin;%PATH%" && echo [OK] FFmpeg added to PATH.

echo [7/8] Installing requirements...
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --quiet >nul 2>&1
if errorlevel 1 (
    echo   First attempt failed. Retrying with output...
    "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 goto :err_requirements
)

:: Verify critical imports
"%PYTHON_EXE%" -c "import fastapi; import uvicorn; import requests; import numpy; import soundfile" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Some required packages failed to install.
    echo   Try running install.bat again, or manually run:
    echo   %PYTHON_EXE% -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Requirements installed and verified.

:: ============================================
:: Step 8: Launch
:: ============================================
echo [8/8] Launching Music Manager...
echo.
echo ============================================
echo.

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" "%SCRIPT_DIR%music_manager.py" %*

if errorlevel 1 (
    echo.
    echo The Music Manager exited with an error.
    echo.
    pause
)

endlocal
goto :eof

:: ============================================
:: Error handlers
:: ============================================
:err_python
echo.
echo ERROR: Failed to download or extract Python.
echo   - Check your internet connection
echo   - URL: %PYTHON_URL%
echo.
pause
exit /b 1

:err_pip
echo ERROR: Failed to install pip.
pause
exit /b 1

:err_requirements
echo.
echo ERROR: Failed to install required packages.
echo   - Check your internet connection
echo   - Try running install.bat again
echo.
pause
exit /b 1
