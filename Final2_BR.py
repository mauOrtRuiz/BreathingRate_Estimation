
import json
import numpy as np
import matplotlib.pyplot as plt
window_sec=15

# Load JSON file
with open("D:/HR2/H10_log_20250611_2133.json", "r") as f:
    data = json.load(f)


# Extract RR, timestamps, and HR
rr_list = []
t_list = []
hr_list = []


# Generate the fields data vectors
for entry in data:
    ts = entry.get("ts")
    rr_values = entry.get("rr", [])
    hr = entry.get("hr", None)

    # Flatten rr values while assigning time
    for rr in rr_values:
        rr_list.append(rr)
        t_list.append(ts)  # or ts + some offset if multiple RRs per timestamp
        hr_list.append(hr)

rr_array = np.array(rr_list)
t_array = np.array(t_list)
hr_array = np.array(hr_list)


# Generate time vector in format HH:MM:SS with 2 decimals
from datetime import datetime
#time_hhmmss = [datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + str(int((ts % 1) *10)) for ts in t_array]
time_hhmmss_hundredths = [
    datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + f"{int((ts % 1) * 100):02d}"
    for ts in t_array
]




import scipy.signal as signal
# Thos does the SIGNAL  PROCESSING

# 1. Generate timestamps based on RR intervals (assuming ms)
t_rr = np.cumsum(rr_array) / 1000.0  # convert to seconds

# 2. Uniform time base at 6 Hz (step = 1/6 s ≈ 0.1667 s)
fs = 6.0
t_uniform = np.arange(t_rr[0], t_rr[-1], 1/fs)

# 3. Interpolate RR onto uniform time grid
rr_uniform = np.interp(t_uniform, t_rr, rr_array)

# 4. Compute and plot PSD (Power Spectral Density)
f_raw, Pxx_raw = signal.welch(rr_uniform, fs=fs, nperseg=256)

plt.figure(figsize=(10, 4))
plt.semilogy(f_raw, Pxx_raw)
plt.title("PSD of Resampled RR Signal (Raw)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Butterworth PBF design
lowcut = 0.2
highcut = 1.2
order = 4
b, a = signal.butter(order, [lowcut / (fs/2), highcut / (fs/2)], btype='band')

# 6. Apply filter
rr_filtered = signal.filtfilt(b, a, rr_uniform)

 # 7. Compute and plot PSD of filtered signal
f_filt, Pxx_filt = signal.welch(rr_filtered, fs=fs, nperseg=256)
plt.figure(figsize=(10, 4))
plt.semilogy(f_filt, Pxx_filt)
plt.title("PSD of Filtered RR Signal (0.2–1.2 Hz)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.show()



# 8. Plot the resampled vs filtered RR signal
plt.figure(figsize=(12, 4))
plt.plot(t_uniform, rr_uniform, label="Original Resampled RR", alpha=0.6)
plt.plot(t_uniform, rr_filtered, label="Filtered RR (0.2–1.2 Hz)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("RR Interval (ms)")
plt.title("Comparison of Resampled and Filtered RR Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





# Use the PSD of the filtered RR signal (from earlier step)
# f_filt: frequency axis
# Pxx_filt: power at each frequency

 
# Limit to the typical breathing frequency band
valid_band = (f_filt >= 0.2) & (f_filt <= 1.2)


# Find frequency with max power in this band
dominant_freq = f_filt[valid_band][np.argmax(Pxx_filt[valid_band])]

 
# Convert to Breathing Rate (breaths per minute)
BR = dominant_freq * 60
print(f"Estimated Breathing Rate (BR): {BR:.2f} breaths per minute")

from scipy.signal import stft

# STFT parameters
nperseg = int(fs * 50)   # window length = 8 seconds
noverlap = int(fs * 8)  # overlap = 6 seconds

 
# Compute STFT
f_stft, t_stft, Zxx = stft(rr_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap)

 
# Plot Spectrogram
plt.figure(figsize=(12, 5))
plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud')
plt.colorbar(label="Magnitude")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Spectrogram of Filtered RR Signal")
plt.axhline(0.2, color='red', linestyle='--', alpha=0.5, label="BR min (0.2 Hz)")
plt.axhline(1.2, color='green', linestyle='--', alpha=0.5, label="BR max (1.2 Hz)")
plt.ylim([0, 2])  # Zoom in on respiratory band
plt.legend()
plt.tight_layout()
plt.show()

# Create uniform time array (if not already)
t_uniform = np.arange(t_rr[0], t_rr[0] + len(rr_filtered)/fs, 1/fs)

# Define time window in seconds
start_sec = 1600
end_sec = 1700

# Create mask to select that window
mask = (t_uniform >= start_sec) & (t_uniform <= end_sec)

# Apply mask to extract section
rr_window = rr_filtered[mask]
t_window = t_uniform[mask]

 
plt.figure(figsize=(12, 4))
plt.plot(t_window, rr_window, label="Filtered RR")
plt.xlabel("Time (s)")
plt.ylabel("RR value")
plt.title("RR Signal from 1600 to 1700 seconds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from scipy.signal import welch
f_win, Pxx_win = welch(rr_window, fs=fs, nperseg=256)
plt.figure(figsize=(10, 4))
plt.semilogy(f_win, Pxx_win)
plt.title("Power Spectral Density (1600–1700s segment)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.axvline(0.2, color='red', linestyle='--', alpha=0.5)
plt.axvline(1.2, color='green', linestyle='--', alpha=0.5)
plt.grid(True)
plt.tight_layout()
plt.show()



 

from scipy.signal import spectrogram
import pandas as pd

 
fs = 6  # sampling frequency

def compute_br(rr_filtered, window_sec):
    nperseg = fs * window_sec
    noverlap = int(nperseg * 0.5)

    f, t_stft, Sxx = spectrogram(rr_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap)
    br_list = []
    timestamp_list = []

    for i, t_val in enumerate(t_stft):
        spectrum_slice = Sxx[:, i]
        band_mask = (f >= 0.2) & (f <= 1.2)
        f_band = f[band_mask]
        P_band = spectrum_slice[band_mask]

 
        if len(P_band) == 0 or np.all(P_band == 0):
            br = np.nan
        else:
            f_dom = f_band[np.argmax(P_band)]
            br = np.round(f_dom * 60, 1)  # Round to 1 decimal

 
        # Convert to HH:MM:SS
        total_seconds = int(t_val)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        timestamp = f"{hh:02d}:{mm:02d}:{ss:02d}"

 
        timestamp_list.append(timestamp)
        br_list.append(br)
        

    return timestamp_list, br_list, t_stft





from datetime import datetime, timedelta

def generate_mirror_from_time_hhmmss(start_str, num_values, step_sec):
    """
    start_str: first timestamp from time_hhmmss_hundredths, format "HH:MM:SS.hh"
    num_values: number of BR points
    step_sec: estimated time between BR points (e.g. 7.5 s for 15s window with 50% overlap)
    """
    # Parse input time string
    base_time = datetime.strptime(start_str, "%H:%M:%S.%f")
    timestamps = []

    for i in range(num_values):
        t = base_time + timedelta(seconds=i * step_sec)
        timestamps.append(t.strftime("%H:%M:%S.") + f"{int(t.microsecond / 10000):02d}")

    return timestamps

ts_s, br_s, t_s = compute_br(rr_filtered, window_sec)


# Use your own variable from earlier
start_time_str = time_hhmmss_hundredths[0]

# Step size depends on STFT settings; use 7.5 if 15s window with 50% overlap

mirror_ts = generate_mirror_from_time_hhmmss(
    start_str=time_hhmmss_hundredths[0],
    num_values=len(br_s),
    step_sec=window_sec * 0.5  # e.g. 7.5 for 15s window with 50% overlap
)


df = pd.DataFrame({
    "Timestamp": mirror_ts,
    "BR_15s (bpm)": br_s
})
df.to_csv("br_Estimated.csv", index=False)

 

# Ensure mirror_ts and br_15s are defined

 

plt.figure(figsize=(14, 5))
plt.scatter(mirror_ts, br_s, color='darkgreen', s=20, label="Breathing Rate (bpm)")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Breathing Rate (bpm)")
plt.title(f"Breathing Rate Variation Over Time ({window_sec}s window)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



        
        
        
#***************************
# ANOTHER TEST


def identify_steady_segments(br_list, timestamp_list, targets, tolerance=0.5, min_duration=3):
    """
    Identifies time segments where BR stays close to any target value for at least `min_duration` samples.
    Returns a list of dictionaries: each with target, start time, end time, and length.
    """
    results = []

    for target in targets:
        in_segment = False
        start_idx = 0

        for i, br in enumerate(br_list):
            if br is not None and abs(br - target) <= tolerance:
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    if i - start_idx >= min_duration:
                        segment = {
                            "target": target,
                            "start_time": timestamp_list[start_idx],
                            "end_time": timestamp_list[i - 1],
                            "length": i - start_idx
                        }
                        results.append(segment)
                    in_segment = False

 
        # Catch if segment goes to the end
        if in_segment and len(br_list) - start_idx >= min_duration:
            segment = {
                "target": target,
                "start_time": timestamp_list[start_idx],
                "end_time": timestamp_list[-1],
                "length": len(br_list) - start_idx
            }
            results.append(segment)

 
    return results


# ***************************************************************
br_targets = [6, 7.5, 9, 12, 15, 30]

# Find steady regions (e.g., min 3 samples ≈ 22.5s with 15s window, 7.5s step)
steady_segments = identify_steady_segments(
    br_list=br_s,
    timestamp_list=mirror_ts,
    targets=br_targets,
    tolerance=1,
    min_duration=3
)
for seg in steady_segments:
    print(f"BR ≈ {seg['target']} bpm from {seg['start_time']} to {seg['end_time']} (length: {seg['length']} samples)")
    
# Convert list of segments into DataFrame

df_segments = pd.DataFrame(steady_segments)

 

# Save to CSV
df_segments.to_csv("br_segments.csv", index=False)
print("Saved steady BR segments to 'br_segments.csv'")

    
    
    
    