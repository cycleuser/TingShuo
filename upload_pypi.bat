@echo off
REM Upload TingShuo to PyPI
REM Prerequisites: pip install build twine

echo === TingShuo PyPI Upload ===

REM Clean previous builds
echo Cleaning old build artifacts ...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"

REM Build
echo Building package ...
python -m build
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

REM Check
echo Checking package ...
python -m twine check dist\*
if errorlevel 1 (
    echo Package check failed!
    pause
    exit /b 1
)

REM Upload
echo Uploading to PyPI ...
python -m twine upload dist\*
if errorlevel 1 (
    echo Upload failed!
    pause
    exit /b 1
)

echo.
echo === Upload complete! ===
echo Install with: pip install tingshuo
pause
