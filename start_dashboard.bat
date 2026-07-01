@echo off
REM Start local web server for dashboard
REM Run this from the TRATA folder: start_dashboard.bat

echo.
echo Starting Dashboard...
echo.
echo Opening http://localhost:8000/dashboard/index.html
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m http.server 8000
