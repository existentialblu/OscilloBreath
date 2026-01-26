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
- Compare APAP chaos vs ASV stability
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

### ResMed-Only Tools (Not Yet Updated):

#### 5. Longitudinal Tracker (`run_longitudinal.bat`)
- Track oscillator metrics over months/years
- See the exact moment ASV stabilized your breathing
- Multiple metrics: flow variability, derivative violence, attractor tightness, etc.
- Smoothed trends to cut through daily noise
- ⚠️ **ResMed EDF only** (Philips support coming soon)

#### 6. Suffering Tracker (`run_suffering_tracker.bat`)
- **The one metric that matters:** Derivative Range
- Track violence of breathing transitions over time
- Color-coded: Red = bad nights, Green = good nights
- Shows your improvement percentage
- ⚠️ **ResMed EDF only** (Philips support coming soon)

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

### Self-Managed ASV Tuning

This tool was built by someone self-managing their ASV after sleep medicine proved useless for UARS and high loop gain ("it's just TeCsA" 🙄).

Key insight from an audio engineer partner: **Unused headroom in your pressure range affects the transfer function.** Tightening PS range to actually-used values (1-7.4 instead of having unused upper range) lets the ASV's control algorithm respond more precisely to your actual breathing dynamics.

### Philips DreamStation 2 Decryption

Philips encrypts DreamStation 2 data with AES-256-GCM to lock users out of their own data. OscilloBreath includes a Python port of [OSCAR](https://www.sleepfiles.com/OSCAR/)'s decryption implementation, which reverse-engineered the encryption in 2022.

**The encryption key is literally:** `"Patient access to their own data"`

The OSCAR developers made a statement with that. We stand with them - your health data belongs to YOU.

**Technical details:** Multi-stage decryption using AES-256-ECB, PBKDF2-SHA256 (10,000 iterations), and AES-256-GCM. Implementation based on OSCAR's `prs1_loader.cpp`.

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
├── philips_loader.py              # Philips DreamStation parser + decryption
├── data_loader.py                 # Universal loader (auto-detects format)
├── run.bat                         # Launch single night analysis
├── run_longitudinal.bat           # Launch longitudinal analysis
├── run_suffering_tracker.bat      # Launch suffering tracker
├── run_single_night.bat           # Launch single night deep dive
├── run_phase_transitions.bat      # Launch transition analyzer
├── run_lyapunov.bat               # Launch Lyapunov analysis
├── run_lyapunov_fast.bat          # Launch fast Lyapunov analysis
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
