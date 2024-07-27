@ECHO OFF

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Python is not installed. Please install Python and try again.
    PAUSE
    EXIT /B
)

REM Check if requirements are already installed
pip show -q -r requirements.txt >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO Requirements are already installed.
    PAUSE
    EXIT /B
)

REM Install requirements
pip install -r requirements.txt

ECHO Requirements have been successfully installed.
PAUSE
