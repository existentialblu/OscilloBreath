@echo off
echo ======================================================
echo OscilloBreath - Longitudinal Bifurcation Analysis
echo Fast batch processing for trend analysis
echo ======================================================
echo.
echo This will process ALL EDF files in a folder.
echo Optimized for speed: ~1 second per night.
echo.
echo Outputs:
echo   - CSV with all metrics per night
echo   - Longitudinal trend visualization
echo   - Correlation analysis between metrics
echo.

python bifurcation_longitudinal.py

echo.
echo Press any key to exit...
pause >nul
