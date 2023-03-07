import matplotlib.pyplot as plt
import scipy.fft as sfft
import customFunctions as cf
import scipy.stats as sstts
import scipy.signal as ssgnl
import pyedflib as edf
import antropy as ant
import pandas as pd
import numpy as np
import pywt
import math
import re
import os

# Reading Info Files, Information Extraction
chb = 'chb05/'
file_index = 1
dir_path = os.path.dirname(os.path.realpath(__file__))
read_path = dir_path + '/EDF Files/' + chb
save_path = dir_path + '/CSV Files/' + chb
os.makedirs(save_path) if not os.path.exists(save_path) else False
edf_files = sorted([file for file in os.listdir(read_path) if file.endswith('.edf')])
txt_file = [file for file in os.listdir(read_path) if file.endswith('.txt')]
summary_file = open(read_path + txt_file[0], 'r')
summary_text = summary_file.read()
seizure_times = np.asarray(re.findall('(?<=Number of Seizures in File: )((?!0)\d{1})', summary_text)).astype(np.int64)
start_times = np.asarray(re.findall('(?<=Start Time: )(.*?)(?= sec)', summary_text)).astype(np.int64)
end_times = np.asarray(re.findall('(?<=End Time: )(.*?)(?= sec)', summary_text)).astype(np.int64)
summary_file.close()
counter = 0

# Reading EDF File
f = edf.EdfReader(read_path + edf_files[file_index])
signal_labels = (f.getSignalLabels())
chn = signal_labels.index('FZ-CZ')
s_freq = int(f.getSampleFrequency(chn))
duration = int(f.getFileDuration())

# Setting Up Variables
interval = 30
epoch_value = math.trunc(duration/interval)
epoch_length = s_freq*interval
last_index = epoch_value*epoch_length
rsl = 4
nperseg = rsl * s_freq

# Seziure Check Column Setup
seizure_check = np.zeros(epoch_value)
for k in range(seizure_times[file_index]):
    st = start_times[file_index + counter]
    et = end_times[file_index + counter]
    st_epoch = math.trunc(st/interval)
    et_ecoch = math.trunc(et/interval) if et % 30 == 0 else math.trunc(et/interval) + 1
    seizure_check[st_epoch:et_ecoch] = 1
ones = np.where(seizure_check == 1)[0]

# Time-Amplitude Data
signal_full = f.readSignal(chn)[0:last_index]
signal_divided = np.split(signal_full, epoch_value)
freqs, psd_full = ssgnl.welch(signal_full, s_freq, nperseg=nperseg)
peak_values = ssgnl.find_peaks(psd_full)
peak_frequency = peak_values[0][1]

_, psd = list(zip(*map(lambda row: ssgnl.welch(row, s_freq, nperseg=nperseg), signal_divided)))

# New Feature Test
new_feature = list(map(lambda row: ant.spectral_entropy(row, sf=s_freq, method='welch'), signal_divided))
ax = np.arange(0, epoch_value)
plt.figure(figsize=(10, 8))
plt.plot(ax, new_feature)
plt.axvspan(ones[0], ones[-1], color='red', alpha=0.5)
plt.show()

'''# Figure: Frequency - Power Spectral Density
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd_full, color='k', lw=2)
plt.xlim([0, freqs.max()])
plt.ylim([0, psd_full.max() * 1.1])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.title("Welch's Periodogram for %s" % edf_files[0])
plt.show()'''

'''#Figure: Time - Frequency - Spectogram
plt.pcolormesh(t, f, 10*np.log10(sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()'''
