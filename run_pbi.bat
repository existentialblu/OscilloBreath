@echo off
echo OscilloBreath - PBI and LF Power Tracker
echo Two focused respiratory stability metrics
echo.
echo PBI = therapy effectiveness (lower = better)
echo LF  = health status (spikes = sick?)
echo.
python "%~dp0pbi_longitudinal.py"
echo.
pause
