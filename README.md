# OscilloBreath

**Coupled Oscillator Analysis for PAP Flow Data**

A tool for analyzing respiratory flow data through the lens of coupled oscillator physics. Born from a late-night nerd snipe about pendulums and breathing patterns.

## What Is This?

Traditional sleep analysis treats each breath as an isolated event. OscilloBreath recognizes that breathing is a **coupled oscillator system** - your respiratory control loops interact with mechanical feedback, creating complex dynamics that show up as high loop gain, periodic breathing, and that lovely phenomenon sleep medicine calls "TeCsA" (while missing the entire control theory aspect).

This tool analyzes flow rate data from ResMed PAP machines to visualize and quantify oscillator behavior, particularly the **violence of breathing transitions** rather than just counting events.

## The Core Insight

**Derivative Range = Suffering**

Traditional metrics count apneas and hypopneas. We measure how *violently* your respiratory oscillator changes state. High derivative range = jerky, chaotic transitions = sympathetic activation = fragmented sleep = you feel like crap.

## Features

### Universal Tools (Support Both ResMed & Philips):

#### 1. Phase Space Analysis (`run.bat`)
- Visualize breathing as a phase space portrait (flow vs dFlow/dt)
- See your respiratory attractor structure
- Color-coded by time to show evolution
- Compare APAP limit cycles vs ASV adaptive chaos
- ✅ **Works with both ResMed and Philips**

#### 2. Single Night Deep Dive (`run_single_night.bat`)
- See *when* during a single night suffering occurred
- 5-minute windowed analysis
- Identifies worst/best periods
- Compare sleep onset, REM periods, settings changes
- ✅ **Works with both ResMed and Philips**

#### 3. Phase Transition Analyzer (`run_phase_transitions.bat`)
- Detect when oscillator changes states
- Get "Transitions/Hour" metric (AHI-like but for state changes)
- Visual markers showing exact transition moments
- Less correlated with suffering than expected, but cool visualization
- ✅ **Works with both ResMed and Philips**

#### 4. Lyapunov Exponent Analysis (`run_lyapunov_fast.bat`)
- **The chaos metric** - Largest Lyapunov Exponent
- Measures fundamental stability vs chaos of the oscillator
- Positive = responsive/adaptive (good!)
- Near-zero = rigid periodic (bad)
- Negative = over-damped (sluggish)
- 3D phase space reconstruction using time-delay embedding
- Optimized version (2-5 minutes per file)
- ✅ **Works with both ResMed and Philips**

#### 5. Longitudinal Lyapunov Tracker (`run_lyapunov_longitudinal.bat`)
- **LIMITED UTILITY**: LLE is very noisy night-to-night, trends are hard to see even with heavy smoothing
- One LLE value per night (concatenates multiple sessions from same date)
- Higher LLE = adaptive/responsive oscillator (GOOD - controlled chaos)
- Lower LLE = rigid/periodic oscillator (BAD - unable to adapt to sleep stage changes)
- **Best use**: Compare specific bad nights vs good nights directly, not for trend tracking
- LLE may be more of a personal respiratory constant than a changing metric
- Use derivative range (suffering tracker) for actual longitudinal trend analysis
- ⚠️ **ResMed EDF only** (uses filename timestamps)

#### 6. Bifurcation Detector (`run_bifurcation.bat`)
- **Predict collapse into periodicity**: Detects when healthy adaptive chaos degrades into rigid limit cycles
- Scans whole night for minimum LLE window (maximum periodicity = "collapse")
- Analyzes the pre-collapse window for predictive signatures
- Computes **Respiratory Reynolds number** candidates — dimensionless ratios predicting regime transition
- Includes IPAP-based metrics when pressure data available (machine's "oh shit" signal)
- ~10 seconds per night on modest hardware
- ✅ **Works with both ResMed and Philips**

#### 7. Longitudinal Bifurcation Analysis (`run_bifurcation_longitudinal.bat`)
- Track bifurcation metrics across months/years of data
- Concatenates multiple sessions per night automatically
- Outputs CSV with all Reynolds candidates for further analysis
- Correlation matrix showing which metrics predict collapse depth
- Finds optimal coefficients for combined Reynolds formulas
- ~1 second per night (processes 1 year of data in ~6 minutes)
- ⚠️ **ResMed EDF only** (uses filename timestamps for night grouping)

### ResMed-Only Tools (Not Yet Updated):

#### 8. Longitudinal Tracker (`run_longitudinal.bat`)
- Track oscillator metrics over months/years
- See the exact moment ASV stabilized your breathing
- Multiple metrics: flow variability, derivative violence, attractor tightness, etc.
- Smoothed trends to cut through daily noise
- ⚠️ **ResMed EDF only** (Philips support coming soon)

#### 9. Suffering Tracker (`run_suffering_tracker.bat`)
- **The one metric that matters:** Derivative Range
- Track violence of breathing transitions over time
- Color-coded: Red = bad nights, Green = good nights
- Shows your improvement percentage
- ⚠️ **ResMed EDF only** (Philips support coming soon)

### Synthetic Data Generator (Web App):

#### 10. Synthetic Data Generator (`synthetic_generator.html`) 🧸 **TOY MODEL**
- **⚠️ Educational toy model - not physiologically accurate, useful for demonstration and testing**
- **Interactive coupled oscillator simulator** for generating synthetic respiratory data
- **Visual 3-segment pendulum** with color-cycling trace (full spectrum every 60 seconds)
- Shows chemical drive, mechanical response, and airway dynamics as coupled segments
- Adjust PALM parameters (Loop gain, Arousal threshold, Pharyngeal collapsibility, Muscle compensation)
- Simulate different sleep stages (N3, REM, N2, wake) with ultradian rhythm cycling
- Add perturbations: flow limitation events, APAP diagnostic puffs (5 Hz FOT at 3/hour)
- Real-time visualization with speed control (0.05x to 0.20x) - slow down to watch individual breaths
- Smart view controls: auto-scrolls after 60s to follow live data, X-axis zoom only (Y-axis locked)
- Batch generation (4 hours of data)
- Loop gain evolution (simulate destabilization over time)
- Export as CSV at 25 Hz, then convert to EDF format using `csv_to_edf.py`
- Pure HTML/JavaScript - no installation needed, just open in browser
- 🧪 **Useful for testing analysis tools and exploring concepts, not for clinical accuracy**

#### 11. CSV to EDF Converter (`run_csv_to_edf.bat`)
- Convert synthetic CSV data to ResMed-compatible EDF format
- Automatically resamples to 25 Hz to match ResMed devices
- Creates proper EDF headers with signal metadata
- Output works with all existing OscilloBreath analysis tools

## Installation

### Requirements
- Python 3.7+
- Windows (batch file launchers, but scripts work cross-platform)

### Setup
```bash
pip install -r requirements.txt
```

Dependencies:
- numpy - Math
- plotly - Interactive plots
- pyedflib - ResMed EDF file reading
- scipy - Signal processing
- cryptography - Philips DreamStation 2 decryption

## Supported Devices

### ResMed
- **AirSense 10 / AirCurve 10** series (APAP, CPAP, ASV)
- Data format: Standard EDF files from SD card
- File pattern: `YYYYMMDD_HHMMSS_BRP.edf`

### Philips Respironics
- **DreamStation** (original)
- **DreamStation 2** (encrypted format - automatically decrypted)
- **System One** series
- Data format: Proprietary binary files (.000, .001, .002, .005, .006)
- Folder-based: All files from SD card session
- **Note:** DreamStation 2 uses AES-256 encryption which is transparently handled

## Usage

### ResMed Users:
1. **Single night:** Run any universal tool `.bat` file and select an EDF file
2. **Longitudinal:** Use `run_longitudinal.bat` or `run_suffering_tracker.bat` and select a folder containing multiple EDF files

### Philips Users:
1. Copy entire SD card contents to a folder (all .000, .001, .002, .005, .006 files)
2. **Single night:** Run any universal tool `.bat` file and select any file in the folder (it will automatically use the parent folder)
3. **Longitudinal:** Not yet supported (use single-night analysis only)

**Note:** Universal tools automatically detect whether you're using ResMed or Philips data. Look for the ✅ icon in the Features section.

### General:
1. **First time:** Run one of the `.bat` files to analyze your data
2. **Output:** All results saved to `output/` folder as interactive HTML
3. **Screenshots:** Click the camera icon in the plot toolbar to save PNG

### Synthetic Data Generator (Toy Model):
1. **Open:** Double-click `synthetic_generator.html` to open in your web browser
2. **Adjust sliders:** Set PALM parameters, noise level, sleep stage, perturbations
3. **Real-time mode:** Click "Start Real-Time Visualization" to watch the oscillator evolve (very slow for observation)
4. **Auto-scroll:** After 60 seconds, view automatically scrolls to show last 60s of data
5. **Manual exploration:** Scroll wheel zooms X-axis, drag to pan, "Follow Live Data" button returns to auto-scroll
6. **Batch mode:** Click "Generate 4-Hour Batch" to create a full night of synthetic data
7. **Export CSV:** Click "Export as CSV" to download raw data
8. **Convert to EDF:** Run `run_csv_to_edf.bat` and select the CSV file to create ResMed-compatible EDF (25 Hz)
9. **Analyze:** Use any OscilloBreath analysis tool on the generated EDF file
10. **⚠️ Remember:** This is a toy model for demonstration - not physiologically accurate, but useful for:
   - Testing analysis tools on known synthetic patterns
   - Exploring coupled oscillator concepts visually
   - Demonstrating how loop gain affects breathing stability
   - Educational purposes and concept exploration

## The Science (Sort Of)

Your respiratory system is a coupled oscillator:
- **Chemical drive** (CO2/O2 sensing)
- **Mechanical response** (breathing muscles)
- **PAP feedback** (machine trying to help/interfere)

High loop gain = these oscillators fight each other, creating beat patterns, waxing/waning, and that characteristic "Cheyne-Stokes respiration" look that traditional medicine treats as isolated "central apneas" instead of recognizing the underlying control system instability.

### Why Derivative Range?

**Flow variability** = amplitude of oscillation (you still breathe big/small on good nights)

**Derivative range** = *violence* of transitions (how hard the system jerks between states)

Violence = arousals = fragmented sleep = suffering.

It's not about the size of the swings, it's about how aggressively you transition between states. Smooth evolution through your attractor = good. Violent jumps = bad.

### Lyapunov Exponents: The Chaos Metric

The **Largest Lyapunov Exponent (LLE)** measures the fundamental nature of your oscillator:

- **Positive LLE**: Responsive/adaptive dynamics - system can respond to changing demands (sleep stages, position, CO2 levels). Indicates flexibility and adaptability.
- **Near-Zero LLE**: Rigid limit cycle - locked into inflexible periodic pattern. Cannot adapt to changing conditions.
- **Negative LLE**: Over-damped - trajectories converge too aggressively, sluggish response.

**Key insight**: LLE and derivative range are *orthogonal* metrics measuring different aspects of breathing quality:
- **LLE** = Adaptability/responsiveness of the oscillator
- **Derivative Range** = Violence/smoothness of transitions

**The goal is positive LLE with low derivative range** - a responsive oscillator that adapts smoothly. Well-tuned ASV shows positive LLE (adaptive) with low derivative violence (smooth). Poorly tuned shows near-zero LLE (rigid) with high derivative violence (jerky).

**Technical details**: Uses time-delay embedding (Takens' theorem) to reconstruct the attractor from 1D flow data, then applies Rosenstein's algorithm to measure trajectory divergence rates.

**Longitudinal note**: LLE appears to be relatively constant night-to-night within an individual, possibly representing a fundamental respiratory phenotype rather than a metric that changes meaningfully over time. High night-to-night variability limits its utility for tracking therapy changes. For longitudinal trend analysis, use derivative range instead.

### Respiratory Reynolds Number: Predicting Collapse

Healthy breathing lives "just into strange" — bounded chaos that allows adaptive response to changing conditions. The **Respiratory Reynolds number** is a dimensionless ratio that predicts when this healthy chaos will collapse into rigid periodicity — the bifurcation point where you get trapped in a limit cycle.

Like the fluid dynamics Reynolds number (inertial/viscous forces), the Respiratory Reynolds captures the ratio of destabilizing to stabilizing forces:

**Best predictor found:**
```
RR = energy_ratio = Σ(dFlow/dt)² / Σ(Flow)²
```

This measures how much of your breathing effort goes into state changes vs actual air movement. High energy_ratio = inefficient oscillation, system "fighting itself" = predicts deeper collapse into periodicity.

**What about IPAP/pressure data?** We tested whether machine pressure response (ipap_trend — is pressure rising?) would add predictive power. Result: IPAP metrics show interesting patterns on individual nights but add essentially zero correlation longitudinally (r ≈ 0.03). The machine's "oh shit" signal predicts *that* collapse is approaching, but not *how deep* it goes. Flow metrics tell you about your breathing; pressure tells you about the machine's opinion, which turns out to be less predictive than your actual breathing dynamics.

## Key Findings (n=1, 455 nights)

Analysis of one user's data from December 2024 to February 2026, spanning APAP and ASV therapy:

### The Bifurcation Framework Works
- **energy_ratio correlates with collapse depth** (r = -0.49 with min_lle)
- Higher pre-collapse energy_ratio → deeper collapse into periodicity
- This is the strongest single predictor found among all candidates tested

### APAP vs ASV: Clear Regime Difference
- **ASV delays collapse 2-3x** compared to APAP (collapse at 400-700 min vs 100-300 min)
- ASV maintains healthy adaptive chaos longer before any periodic dip
- The APAP → ASV transition (Feb 2025) is clearly visible in all metrics
- ASV creates actual bistability; APAP keeps the system in a gray zone

### Good Nights vs Bad Nights
- **Good nights have shallower collapses** — min_lle stays higher (0.04-0.05 vs 0.02-0.03)
- Good nights show cleaner separation between pre-collapse and stable-chaos states
- Bad nights: the system is always kind of approaching collapse, just sometimes it gets there
- Good nights: distinct stable regime, so approaching collapse looks different from baseline

### What Doesn't Predict Collapse Depth
- **derivative_violence**: r ≈ 0 (violence of breathing doesn't predict how deep you fall)
- **cv_ratio**: r ≈ 0 (variability of breathing rate relative to depth)
- **IPAP metrics**: r ≈ 0 longitudinally (machine response doesn't predict collapse severity)

### Implications for Therapy
- Energy_ratio in the pre-collapse window may be a useful metric for tuning
- ASV's adaptive pressure support appears to genuinely stabilize the oscillator
- The goal is not to eliminate all periodicity, but to keep collapses shallow (stay "just into strange")

### Self-Managed ASV Tuning

This tool was built by someone self-managing their ASV after sleep medicine proved useless for UARS and high loop gain ("it's just TeCsA" 🙄).

Key insight from an audio engineer partner: **Unused headroom in your pressure range affects the transfer function.** Tightening PS range to actually-used values (1-7.4 instead of having unused upper range) lets the ASV's control algorithm respond more precisely to your actual breathing dynamics.

### Philips DreamStation 2 Compatibility

DreamStation 2 uses encrypted data files. OscilloBreath includes a Python implementation of the file format parser originally developed by the [OSCAR](https://www.sleepfiles.com/OSCAR/) project, which added DreamStation 2 support in 2022.

This allows users to access their own therapy data for personal analysis, consistent with the principle that patients should have access to their own health information.

**Technical details:** File format uses AES-256-GCM encryption with PBKDF2-SHA256 key derivation. Implementation based on OSCAR's `prs1_loader.cpp`.

## File Structure

```
OscilloBreath/
├── analyze.py                      # Single night phase space
├── longitudinal_analysis.py        # Multi-night metrics
├── suffering_tracker.py            # The one metric that matters
├── single_night_suffering.py       # When suffering occurred
├── phase_transition_analyzer.py    # State change detection
├── lyapunov_analyzer.py           # Chaos analysis (LLE)
├── lyapunov_analyzer_fast.py      # Optimized version with downsampling
├── lyapunov_longitudinal.py       # Longitudinal LLE tracker
├── bifurcation_detector.py        # Find collapse points, compute Reynolds numbers
├── bifurcation_longitudinal.py    # Longitudinal bifurcation analysis
├── philips_loader.py              # Philips DreamStation parser + decryption
├── data_loader.py                 # Universal loader (auto-detects format, now with pressure)
├── synthetic_generator.html       # Interactive web app for synthetic data
├── csv_to_edf.py                  # Convert synthetic CSV to ResMed EDF format
├── run.bat                         # Launch single night analysis
├── run_longitudinal.bat           # Launch longitudinal analysis
├── run_suffering_tracker.bat      # Launch suffering tracker
├── run_single_night.bat           # Launch single night deep dive
├── run_phase_transitions.bat      # Launch transition analyzer
├── run_lyapunov.bat               # Launch Lyapunov analysis
├── run_lyapunov_fast.bat          # Launch fast Lyapunov analysis
├── run_lyapunov_longitudinal.bat  # Launch longitudinal LLE tracker
├── run_bifurcation.bat            # Launch bifurcation detector
├── run_bifurcation_longitudinal.bat # Launch longitudinal bifurcation analysis
├── run_csv_to_edf.bat             # Launch CSV to EDF converter
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── README.txt                     # Plain text instructions
├── LICENSE                        # MIT License
└── output/                        # Generated plots (not in git)
```

## Credits

Built during a late-night nerd snipe session after watching a video about coupled pendulums.

Inspired by:
- Coupled oscillator physics
- Frustration with sleep medicine's event-counting obsession
- An audio engineer's insight about transfer functions and headroom
- The realization that SampleEntropy was circling around the problem but derivative range *is* the problem

## License

MIT License - Do whatever you want with this. If it helps you understand your breathing, awesome. If you use it to optimize your PAP settings, even better. If you show it to a sleep doctor and they finally understand loop gain, please record their reaction.

## Disclaimer

This is a data analysis tool for personal exploration. It is not medical advice. I am not a doctor. I am someone with UARS and high loop gain who got tired of being told "it's just TeCsA" and decided to treat their breathing as the coupled oscillator system it actually is.

Self-management of PAP therapy should be done carefully and ideally with medical oversight (even if that oversight is useless for UARS/loop gain, at least have someone checking you're not doing something actively harmful).

---

*"The derivative range puts my earlier efforts with SampleEntropy to shame."*
