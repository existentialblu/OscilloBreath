@echo off
echo ======================================================
echo OscilloBreath - FAST Lyapunov Exponent Analysis
echo ======================================================
echo.
echo Optimized version with heavy downsampling
echo Should complete in 2-5 minutes
echo.
echo Measuring chaos in your respiratory oscillator:
echo   Positive LLE = Chaotic (unstable)
echo   Zero LLE = Periodic (neutral)
echo   Negative LLE = Stable (convergent)
echo.

python lyapunov_analyzer_fast.py

echo.
echo Press any key to exit...
pause >nul
