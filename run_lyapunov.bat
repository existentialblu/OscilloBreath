@echo off
echo ======================================================
echo OscilloBreath - Lyapunov Exponent Analysis
echo ======================================================
echo.
echo WARNING: This analysis is COMPUTATIONALLY INTENSIVE
echo It may take 5-15 minutes depending on file size
echo.
echo Measuring chaos in your respiratory oscillator:
echo   Positive LLE = Chaotic (unstable)
echo   Zero LLE = Periodic (neutral)
echo   Negative LLE = Stable (convergent)
echo.
pause
echo.

python lyapunov_analyzer.py

echo.
echo Press any key to exit...
pause >nul
