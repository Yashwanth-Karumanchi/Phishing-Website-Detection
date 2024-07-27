@echo off
python -c "print('Python is available')"
if errorlevel 1 goto :python_not_found

@python.exe -m streamlit run home.py %*
goto :end

:python_not_found
echo Python is not installed on this system.
pause

:end
