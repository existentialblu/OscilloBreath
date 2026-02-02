@echo off
echo ======================================================
echo OscilloBreath - Bifurcation Detector
echo Finding the Edge of Chaos
echo ======================================================
echo.
echo This tool finds where adaptive breathing collapses into
echo rigid limit cycles, and looks for predictive signatures.
echo.
echo What it does:
echo   1. Scans whole night for minimum LLE (maximum periodicity)
echo   2. Extracts the window BEFORE that collapse
echo   3. Computes candidate "Respiratory Reynolds" numbers
echo   4. Compares pre-collapse vs stable-chaos windows
echo.
echo WARNING: This is computationally expensive.
echo Expect 10-30 minutes for a full night of data.
echo.

python bifurcation_detector.py

echo.
echo Press any key to exit...
pause >nul
