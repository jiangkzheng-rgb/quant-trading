@echo off
chcp 65001 >nul
echo.
echo  运行多因子趋势信号系统...
echo.
cd /d "%~dp0"
python strategy.py
echo.
echo  完成！请刷新浏览器中的 trading_terminal.html
pause
