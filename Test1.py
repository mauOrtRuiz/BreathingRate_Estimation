# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:24:21 2025

@author: UVMInstaller
"""



import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON file
with open("D:/HR2/H10_log_20250611_2133.json", "r") as f:
    data = json.load(f)


# Extract RR, timestamps, and HR
rr_list = []
t_list = []
hr_list = []


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

from datetime import datetime
#time_hhmmss = [datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + str(int((ts % 1) *10)) for ts in t_array]
time_hhmmss_hundredths = [
    datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + f"{int((ts % 1) * 100):02d}"
    for ts in t_array
]

# Optional: sort by timestamp if needed
#sorted_idx = np.argsort(t_array)
#rr_array = rr_array[sorted_idx]
#t_array = t_array[sorted_idx]

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

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



# Plot the resampled vs filtered RR signal
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
import numpy as np
import matplotlib.pyplot as plt
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

ts_10s, br_10s, t_10s = compute_br(rr_filtered, 30)
ts_20s, br_20s, t_20s = compute_br(rr_filtered, 40)

 
# Compute for 10s and 20s
# Step 1: Create the DataFrames first
df_10s = pd.DataFrame({
    "Timestamp": ts_10s,
    "BR_10s (bpm)": br_10s
})

 
df_20s = pd.DataFrame({
    "Timestamp": ts_20s,
    "BR_20s (bpm)": br_20s
})

 
# Step 2: Format BR columns to one decimal (as string)
df_10s["BR_10s (bpm)"] = df_10s["BR_10s (bpm)"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
df_20s["BR_20s (bpm)"] = df_20s["BR_20s (bpm)"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

 

# Step 3: Save to CSV
df_10s.to_csv("br_30s_window.csv", index=False)
df_20s.to_csv("br_40s_window.csv", index=False)


# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(t_10s, br_10s, label="BR from 10s window", marker='o', alpha=0.7)
plt.plot(t_20s, br_20s, label="BR from 20s window", marker='x', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Breathing Rate (bpm)")
plt.title("Comparison of BR estimation (10s vs 20s STFT windows)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

