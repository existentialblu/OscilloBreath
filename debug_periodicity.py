"""
Debug script: Show LLE distribution for a single night
to calibrate the periodicity threshold.
"""

import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
from data_loader import load_flow_data
from bifurcation_longitudinal import compute_windowed_lle_fast

def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select EDF file",
        filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path


def main():
    print("=" * 60)
    print("LLE Distribution Diagnostic")
    print("=" * 60)

    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    print(f"\nLoading: {Path(file_path).name}")
    flow, sample_rate, _ = load_flow_data(file_path)

    duration_hours = len(flow) / sample_rate / 3600
    print(f"Duration: {duration_hours:.2f} hours")

    print("\nComputing windowed LLE (10-min windows)...")
    lle_data = compute_windowed_lle_fast(flow, sample_rate, window_seconds=600)

    lle_values = lle_data['lle_values']
    times_min = lle_data['window_centers_sec'] / 60

    valid_lle = lle_values[~np.isnan(lle_values)]

    print(f"\nWindows computed: {len(lle_values)}")
    print(f"Valid windows: {len(valid_lle)}")

    print("\n" + "=" * 60)
    print("LLE DISTRIBUTION")
    print("=" * 60)
    print(f"  Min:    {np.min(valid_lle):.4f}")
    print(f"  5th %:  {np.percentile(valid_lle, 5):.4f}")
    print(f"  10th %: {np.percentile(valid_lle, 10):.4f}")
    print(f"  25th %: {np.percentile(valid_lle, 25):.4f}")
    print(f"  Median: {np.median(valid_lle):.4f}")
    print(f"  75th %: {np.percentile(valid_lle, 75):.4f}")
    print(f"  Max:    {np.max(valid_lle):.4f}")
    print(f"  Std:    {np.std(valid_lle):.4f}")

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    # Test various thresholds
    thresholds = [0.035, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    for thresh in thresholds:
        count = np.sum(valid_lle < thresh)
        pct = 100.0 * count / len(valid_lle)
        print(f"  LLE < {thresh:.3f}: {count:3d} windows = {pct:5.1f}%")

    # Percentile-based thresholds
    print("\nPercentile-based:")
    for pct in [10, 20, 25, 33]:
        thresh = np.percentile(valid_lle, pct)
        print(f"  {pct}th percentile = {thresh:.4f}")

    print("\n" + "=" * 60)
    print("WINDOW-BY-WINDOW LLE VALUES")
    print("=" * 60)
    for i, (t, lle) in enumerate(zip(times_min, lle_values)):
        if np.isnan(lle):
            print(f"  {t:6.1f} min: NaN")
        else:
            bar = "*" * int(lle * 50)
            print(f"  {t:6.1f} min: {lle:.4f} {bar}")

    print("\n" + "=" * 60)
    print("COLLAPSE DYNAMICS")
    print("=" * 60)

    # Find minimum LLE (skip first window for sleep onset weirdness)
    valid_mask = ~np.isnan(lle_values)
    if times_min[0] < 10:
        valid_mask[0] = False

    valid_indices = np.where(valid_mask)[0]
    min_idx = valid_indices[np.argmin(lle_values[valid_mask])]

    min_lle = lle_values[min_idx]
    min_time = times_min[min_idx]

    print(f"Minimum LLE: {min_lle:.4f} at {min_time:.1f} min")

    # Find IMMEDIATE local maximum before minimum
    # Walk backwards from min until we find a local peak (value higher than neighbors)
    local_max_idx = None
    local_max_lle = None

    for i in range(min_idx - 1, 0, -1):
        if np.isnan(lle_values[i]):
            continue
        # Check if this is a local max (higher than previous)
        prev_valid = None
        for j in range(i - 1, -1, -1):
            if not np.isnan(lle_values[j]):
                prev_valid = lle_values[j]
                break

        if prev_valid is not None and lle_values[i] > prev_valid and lle_values[i] > lle_values[min_idx]:
            # This could be our local max - but keep going if the previous was higher
            if local_max_lle is None or lle_values[i] > local_max_lle:
                local_max_idx = i
                local_max_lle = lle_values[i]
            # Stop once we've found a peak and started descending from it
            if local_max_lle is not None and lle_values[i] < local_max_lle * 0.9:
                break

    # Simpler approach: find the highest point in the 3 windows immediately before min
    lookback = min(5, min_idx)
    if lookback > 0:
        recent_window = lle_values[min_idx - lookback:min_idx]
        recent_times = times_min[min_idx - lookback:min_idx]
        valid_recent = ~np.isnan(recent_window)

        if np.any(valid_recent):
            recent_max_idx = np.argmax(np.where(valid_recent, recent_window, -np.inf))
            immediate_max_lle = recent_window[recent_max_idx]
            immediate_max_time = recent_times[recent_max_idx]

            immediate_descent_min = min_time - immediate_max_time
            immediate_descent_sec = immediate_descent_min * 60

            print(f"\nImmediate pre-collapse max (within {lookback} windows):")
            print(f"  LLE: {immediate_max_lle:.4f} at {immediate_max_time:.1f} min")
            print(f"  >>> IMMEDIATE DESCENT: {immediate_descent_min:.1f} min ({immediate_descent_sec:.0f} sec)")

    # Also show global max for context
    pre_collapse_lle = lle_values[:min_idx + 1]
    pre_collapse_times = times_min[:min_idx + 1]
    valid_pre = ~np.isnan(pre_collapse_lle)

    if np.any(valid_pre):
        pre_valid_indices = np.where(valid_pre)[0]
        max_pre_idx = pre_valid_indices[np.argmax(pre_collapse_lle[valid_pre])]

        max_lle = pre_collapse_lle[max_pre_idx]
        max_time = pre_collapse_times[max_pre_idx]

        global_descent_min = min_time - max_time

        print(f"\nGlobal pre-collapse max:")
        print(f"  LLE: {max_lle:.4f} at {max_time:.1f} min")
        print(f"  Total descent: {global_descent_min:.1f} min ({global_descent_min * 60:.0f} sec)")
        print(f"  LLE drop: {max_lle:.4f} -> {min_lle:.4f} (delta = {max_lle - min_lle:.4f})")

    # Count periodic episodes
    print("\n" + "=" * 60)
    print("PERIODIC EPISODES (LLE < 0.15)")
    print("=" * 60)

    threshold = 0.15
    in_episode = False
    episodes = []
    current_episode_start = None

    for i, (t, lle) in enumerate(zip(times_min, lle_values)):
        if np.isnan(lle):
            continue
        if lle < threshold:
            if not in_episode:
                in_episode = True
                current_episode_start = t
        else:
            if in_episode:
                episodes.append((current_episode_start, t - 10))  # 10-min windows
                in_episode = False

    if in_episode:
        episodes.append((current_episode_start, times_min[-1]))

    print(f"Number of periodic episodes: {len(episodes)}")
    for i, (start, end) in enumerate(episodes):
        duration = end - start + 10  # Include the window duration
        print(f"  Episode {i+1}: {start:.0f}-{end+10:.0f} min (duration: {duration:.0f} min)")

    total_periodic_time = sum(e[1] - e[0] + 10 for e in episodes)
    total_time = times_min[-1] - times_min[0] + 10
    pct_periodic = 100 * total_periodic_time / total_time if total_time > 0 else 0
    print(f"\nTotal time in periodicity: {total_periodic_time:.0f} min ({pct_periodic:.1f}%)")

    print("\n" + "=" * 60)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 60)

    # Use bottom 25th percentile as "periodic"
    p25 = np.percentile(valid_lle, 25)
    median = np.median(valid_lle)

    print(f"Suggested threshold: {p25:.3f} (25th percentile)")
    print(f"This would classify ~25% of windows as 'periodic'")
    print(f"Alternatively, use {median - np.std(valid_lle):.3f} (median - 1 std)")


if __name__ == "__main__":
    main()
