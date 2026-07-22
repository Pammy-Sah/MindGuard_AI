@echo off
title MindGuard AI – Starting...
echo.
echo  ============================================================
echo   MindGuard AI ^| IBM watsonx.ai ^| Granite Models
echo  ============================================================
echo.
echo  [1/2] Installing dependencies...
pip install flask ibm-watsonx-ai PyPDF2 python-dotenv
echo.
echo  [2/2] Starting MindGuard AI server...
echo.
python app.py
pause
