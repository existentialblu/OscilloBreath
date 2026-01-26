OscilloBreath - Coupled Oscillator Breathing Analysis
======================================================

A tool for exploring PAP flow data through the lens of coupled oscillator physics.

FIRST TIME SETUP:
-----------------
1. Open command prompt in this folder
2. Run: pip install -r requirements.txt
   (This installs the needed Python libraries)

HOW TO USE:
-----------

SINGLE NIGHT ANALYSIS (run.bat):
1. Double-click "run.bat"
2. Pick an .edf file when the dialog opens
3. Wait for analysis (might take a minute for large files)
4. Interactive plot opens in your browser automatically
5. Results saved in the "output" folder

LONGITUDINAL ANALYSIS (run_longitudinal.bat):
1. Double-click "run_longitudinal.bat"
2. Pick a FOLDER containing multiple .edf files
3. Script automatically processes all files (>30 min only)
4. See how your oscillator metrics changed over time
5. Results saved to "output/longitudinal_analysis.html"

SUFFERING TRACKER (run_suffering_tracker.bat):
1. Double-click "run_suffering_tracker.bat"
2. Pick a FOLDER containing multiple .edf files
3. Tracks the ONE metric that correlates with actual suffering
4. Derivative Range = violence of breathing changes
5. Color-coded: Red = bad nights, Green = good nights
6. Results saved to "output/suffering_tracker.html"

PHASE TRANSITION ANALYZER (run_phase_transitions.bat):
1. Double-click "run_phase_transitions.bat"
2. Pick a SINGLE .edf file to analyze
3. Detects when your oscillator changes states
4. Gives you "Transitions/Hour" - an AHI-like metric
5. Shows exactly WHEN transitions happen during the night
6. Results saved to "output/[filename]_phase_transitions.html"

SINGLE NIGHT SUFFERING (run_single_night.bat):
1. Double-click "run_single_night.bat"
2. Pick a SINGLE .edf file to analyze
3. Shows WHEN during the night suffering was worst/best
4. 5-minute windowed derivative range over time
5. Identifies worst and best periods of the night
6. Results saved to "output/[filename]_single_night_suffering.html"

LYAPUNOV EXPONENT ANALYSIS (run_lyapunov.bat):
1. Double-click "run_lyapunov.bat"
2. Pick a SINGLE .edf file to analyze
3. WARNING: Takes 5-15 minutes (computationally intensive!)
4. Measures chaos in your respiratory oscillator
5. Positive LLE = chaotic, Zero = periodic, Negative = stable
6. Results saved to "output/[filename]_lyapunov.html"

WHAT YOU'RE SEEING:
-------------------
Top plot (Phase Space):
- X-axis: Flow rate at each moment
- Y-axis: How fast flow is changing
- Color: Time progression (purple=early, yellow=late)
- Stable breathing = tight orbits
- Periodic breathing/loop gain = wild swirls

Middle plot: Raw flow over time
Bottom plot: Rate of change over time

INTERACTION:
------------
- Click and drag to pan
- Scroll to zoom
- Double-click to reset view
- Hover for values
- Use toolbar on right for more options
- 📷 CAMERA BUTTON: Click to download high-res PNG screenshot
  (Great for comparing different nights side-by-side!)

Compare your APAP nights to ASV nights - you should see totally different
orbital structures as the control system stabilizes!

LONGITUDINAL METRICS EXPLAINED:
--------------------------------
Flow Variability: How much your breathing amplitude varies (lower = more stable)
Derivative Variability: How violently flow changes (lower = smoother)
Phase Space Area: Size of the attractor blob (smaller = more constrained)
Attractor Tightness: 1=tight cluster, 0=scattered (higher = better)
Flow/Derivative Range: Peak-to-peak extremes
Aspect Ratio: Elongation of attractor (closer to 1 = more circular/stable)
Stability Score: Combined metric (LOWER is better, more stable breathing)

Watch these metrics change as you move from APAP → poorly tuned ASV → well tuned ASV!

SUFFERING TRACKER EXPLAINED:
----------------------------
Derivative Range = Peak-to-peak change in flow rate
- Measures how VIOLENTLY your breathing transitions between states
- High values = gasping, abrupt stops, respiratory system jerking around
- Low values = smooth, controlled transitions
- This metric directly correlates with sleep fragmentation and subjective suffering

Why it matters more than SampleEntropy or other complexity measures:
- DIRECT measure of violence (not indirect complexity)
- Shows the actual forces your body experiences
- Captures sympathetic nervous system activation
- Correlates with arousals and that "fighting for breath" feeling

The plot shows:
- Color-coded dots (red = suffering, green = good)
- 7-day smoothed trend line
- Overall average vs recent 30-day average
- Stats box showing your improvement percentage

PHASE TRANSITION ANALYZER EXPLAINED:
------------------------------------
Transitions/Hour = Number of times the oscillator changes state per hour

What is a phase transition?
- Your respiratory oscillator exists in different "states" (stable, unstable, chaotic, etc.)
- A transition = moment when the oscillator shifts from one state to another
- Detected by tracking changes in breathing amplitude, variability, and derivative behavior

Why this matters:
- Traditional AHI counts apneas/hypopneas (individual events)
- Transitions/Hour counts STATE CHANGES (oscillator instability)
- Captures loop gain behavior in a way "event counters" can understand
- Lower = more stable oscillator = better sleep

The visualization shows:
- Phase space with red X marks at each transition
- Flow and derivative waveforms with transition markers
- State vector showing oscillator behavior over time
- Exact timing of each transition

Compare APAP vs ASV nights - you'll see way more transitions on unstable nights.

SINGLE NIGHT SUFFERING EXPLAINED:
----------------------------------
This tool shows you WHEN during a single night the oscillator was most violent.

What it shows:
- Phase space colored by violence (red = high, green = low)
- Derivative range calculated in 5-minute sliding windows
- Time-series plot showing suffering score throughout the night
- Red X marks the worst 5-minute period
- Green circle marks the best 5-minute period

Use this to:
- See if suffering happens at sleep onset, during certain hours, etc.
- Identify patterns (early night instability, REM-related issues)
- Compare specific time periods between APAP and ASV nights
- Find what time of night your settings work best/worst

Example: "My APAP has high suffering scores in the first 2 hours (sleep onset),
but my ASV keeps it stable throughout."

LYAPUNOV EXPONENT EXPLAINED:
-----------------------------
The Largest Lyapunov Exponent (LLE) measures chaos in dynamical systems.

What it measures:
- How quickly nearby trajectories in phase space diverge
- Sensitivity to initial conditions
- The fundamental "chaotic-ness" of your oscillator

Interpretation:
- Positive LLE = CHAOTIC - Small perturbations grow exponentially, unpredictable
- Zero LLE = PERIODIC - Limit cycle behavior, predictable oscillation
- Negative LLE = STABLE - Trajectories converge, well-damped system

How it's calculated:
- Reconstructs your respiratory attractor using time-delay embedding (Takens' theorem)
- Tracks how nearby points diverge over time (Rosenstein method)
- Measures exponential divergence rate

Why it matters:
- More fundamental than derivative range
- Captures the NATURE of the oscillator (chaotic vs stable)
- Directly relates to predictability and control
- Compare APAP (expect positive LLE) vs ASV (expect lower/negative LLE)

Note: This is the most computationally intensive analysis (takes 5-15 minutes).

Have fun exploring!
