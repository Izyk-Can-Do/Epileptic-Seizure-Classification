import os
import re
import pywt
import math
import numpy as np
import antropy as ant
import pyedflib as edf
import scipy.signal as ssgnl
import scipy.stats as sstts
import eeglib.features as eegf
import customFunctions as cf

# Reading Info Files, Information Extraction
chb = 'chb22/'
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

for i in range(len(edf_files)):

    # Reading EDF File
    f = edf.EdfReader(read_path + edf_files[i])
    signal_labels = (f.getSignalLabels())
    chn = signal_labels.index('FZ-CZ')
    s_freq = int(f.getSampleFrequency(chn))
    duration = int(f.getFileDuration())

    # Setting Up Variables
    interval = 10
    epoch_value = math.trunc(duration/interval)
    epoch_lenght = s_freq*interval
    last_index = epoch_value*epoch_lenght
    rsl = 4  # Resolution of each frequency bin
    nperseg = rsl * s_freq  #  Length of each segment
    nyq_freq = s_freq/2

    # Time-Amplitude Data
    signal_full = f.readSignal(chn)[0:last_index]
    signal_divided = np.split(signal_full, epoch_value)

    # Peak Frequency
    _, psd_full = ssgnl.welch(signal_full, s_freq, nperseg=nperseg)
    peak_values = ssgnl.find_peaks(psd_full)
    peak_frequency = peak_values[0][1]

    # Frequency - Power Spectral Density Data
    _, psd = list(zip(*map(lambda row: ssgnl.welch(row, s_freq, nperseg=nperseg), signal_divided)))

    # Time - Frequency - Spectogram
    _, _, sxx = list(zip(*map(lambda row: ssgnl.spectrogram(row, s_freq, nperseg=nperseg), signal_divided)))

    # Seziure Check Column Setup
    seizure_check = np.zeros(epoch_value)
    for k in range(seizure_times[i]):
        st = start_times[k+counter]
        et = end_times[k+counter]
        st_epoch = math.trunc(st/interval)
        et_ecoch = math.trunc(et/interval) if et % interval == 0 else math.trunc(et/interval) + 1
        seizure_check[st_epoch:et_ecoch] = 1
    counter += seizure_times[i]

    # Time Domain Feature Extraction
    peak_amplitude = list(map(lambda row: np.amax(row), signal_divided))
    peak_to_peak_amplitude = list(map(lambda row: np.ptp(row), signal_divided))
    rms = list(map(lambda row: np.sqrt(np.mean(np.power(row, 2))), signal_divided))
    line_length = list(map(lambda row: np.mean(np.abs(np.diff(row))), signal_divided))
    quantile = list(map(lambda row: np.quantile(row, q=0.75), signal_divided))
    min_amplitude = list(map(lambda row: np.amin(row), signal_divided))
    zcr = list(map(lambda row: ant.num_zerocross(row), signal_divided))
    energy = list(map(lambda row: pow(sum(abs(row)), 2), signal_divided))
    variance = list(map(lambda row: np.var(row), signal_divided))
    std_dvt = list(map(lambda row: np.std(row, dtype=np.float64), signal_divided))
    median = list(map(lambda row: np.median(row), signal_divided))
    mean = list(map(lambda row: np.mean(row), signal_divided))
    hjort_acivity = list(map(lambda row: eegf.hjorthActivity(row), signal_divided))
    hjort_mobility, hjort_complexity = list(zip(*map(lambda row: ant.hjorth_params(row), signal_divided)))
    skewness = list(map(lambda row: sstts.skew(row), signal_divided))
    kurtosis = list(map(lambda row: sstts.kurtosis(row), signal_divided))
    fano_factor = list(map(lambda row: np.var(row)/np.mean(row), signal_divided))
    form_factor = list(map(lambda row: np.sqrt(np.mean(row**2))/np.mean(row), signal_divided))
    pulse_indicator = list(map(lambda row: np.max(np.abs(row))/np.mean(row), signal_divided))
    total_variation = list(map(lambda row: np.sum(np.abs(np.diff(row))), signal_divided))

    # Frequency Domain Feature Extraction
    band_power_delta = list(map(lambda row: np.amax(row[1*rsl:4*rsl]), psd))
    band_power_theta = list(map(lambda row: np.amax(row[4*rsl:8*rsl]), psd))
    band_power_alpha = list(map(lambda row: np.amax(row[8*rsl:12*rsl]), psd))
    band_power_beta = list(map(lambda row: np.amax(row[12*rsl:30*rsl]), psd))
    line_length_delta = list(map(lambda row: np.mean(np.abs(np.diff(row[1*rsl:4*rsl]))), psd))
    line_length_theta = list(map(lambda row: np.mean(np.abs(np.diff(row[4*rsl:8*rsl]))), psd))
    line_length_alpha = list(map(lambda row: np.mean(np.abs(np.diff(row[8*rsl:12*rsl]))), psd))
    line_length_beta = list(map(lambda row: np.mean(np.abs(np.diff(row[12*rsl:30*rsl]))), psd))
    quantile_delta = list(map(lambda row: np.quantile(row[1*rsl:4*rsl], q=0.75), psd))
    quantile_theta = list(map(lambda row: np.quantile(row[4*rsl:8*rsl], q=0.75), psd))
    quantile_alpha = list(map(lambda row: np.quantile(row[8*rsl:12*rsl], q=0.75), psd))
    quantile_beta = list(map(lambda row: np.quantile(row[12*rsl:30*rsl], q=0.75), psd))
    mean_delta = list(map(lambda row: np.mean(row[1*rsl:4*rsl]), psd))
    mean_theta = list(map(lambda row: np.mean(row[4*rsl:8*rsl]), psd))
    mean_alpha = list(map(lambda row: np.mean(row[8*rsl:12*rsl]), psd))
    mean_beta = list(map(lambda row: np.mean(row[12*rsl:30*rsl]), psd))
    median_delta = list(map(lambda row: np.median(row[1*rsl:4*rsl]), psd))
    median_theta = list(map(lambda row: np.median(row[4*rsl:8*rsl]), psd))
    median_alpha = list(map(lambda row: np.median(row[8*rsl:12*rsl]), psd))
    median_beta = list(map(lambda row: np.median(row[12*rsl:30*rsl]), psd))
    std_dvt_delta = list(map(lambda row: np.std(row[1*rsl:4*rsl], dtype=np.float64), psd))
    std_dvt_theta = list(map(lambda row: np.std(row[4*rsl:8*rsl], dtype=np.float64), psd))
    std_dvt_alpha = list(map(lambda row: np.std(row[8*rsl:12*rsl], dtype=np.float64), psd))
    std_dvt_beta = list(map(lambda row: np.std(row[12*rsl:30*rsl], dtype=np.float64), psd))
    var_delta = list(map(lambda row: np.var(row[1*rsl:4*rsl]), psd))
    var_theta = list(map(lambda row: np.var(row[4*rsl:8*rsl]), psd))
    var_alpha = list(map(lambda row: np.var(row[8*rsl:12*rsl]), psd))
    var_beta = list(map(lambda row: np.var(row[12*rsl:30*rsl]), psd))
    cA = list(map(lambda row: pywt.wavedec(row[0:128*rsl], 'db1')[0], psd))
    cD = list(map(lambda row: pywt.wavedec(row[0:128*rsl], 'db1')[1], psd))

    # Time-Frequency Feature Extraction
    mean_tf = list(map(lambda row: np.std(10*np.log10(row)[peak_frequency]), sxx))
    quantile_tf = list(map(lambda row: np.quantile(10*np.log10(row)[peak_frequency], q=0.75), sxx))
    total_variation_tf = list(map(lambda row: np.sum(np.abs(np.diff(row[peak_frequency]))), sxx))

    # Non-Linear Feature Extraction
    dfa = list(map(lambda row: ant.detrended_fluctuation(row), signal_divided))
    pfd = list(map(lambda row: ant.petrosian_fd(row), signal_divided))
    hfd = list(map(lambda row: ant.higuchi_fd(row), signal_divided))
    kfd = list(map(lambda row: ant.katz_fd(row), signal_divided))
    lziv_complexity = list(map(lambda row: ant.lziv_complexity(row), signal_divided))

    # Entropy Feature Extraction
    shannon_entropy = list(map(lambda row: cf.shn_entropy(row), signal_divided))
    permutation_entropy = list(map(lambda row: ant.perm_entropy(row), signal_divided))
    spectral_entropy = list(map(lambda row: ant.spectral_entropy(row, sf=s_freq, method='welch'), signal_divided))
    svd_entropy = list(map(lambda row: ant.svd_entropy(row), signal_divided))
    app_entropy = list(map(lambda row: ant.app_entropy(row), signal_divided))
    sample_entropy = list(map(lambda row: ant.sample_entropy(row), signal_divided))

    # Headers Row Setup
    headers_array = (
        "seizure_check", "peak_amplitude", "peak_to_peak_amplitude", "rms", "line_length", "quantile", "min_amplitude",
        "zcr", "energy", "variance", "std_dvt", "median", "mean", "hjort_acivity", "hjort_mobility", "hjort_complexity",
        "skewness", "kurtosis", "fano_factor", "form_factor", "pulse_indicator", "total_variation", "band_power_delta",
        "band_power_theta", "band_power_alpha", "band_power_beta", "line_length_delta", "line_length_theta",
        "line_length_alpha", "line_length_beta", "quantile_delta", "quantile_theta", "quantile_alpha", "quantile_beta",
        "mean_delta", "mean_theta", "mean_alpha", "mean_beta", "median_delta", "median_theta", "median_alpha",
        "median_beta", "std_dvt_delta", "std_dvt_theta", "std_dvt_alpha", "std_dvt_beta", "var_delta", "var_theta",
        "var_alpha", "var_beta", "cA", "cD", "mean_tf", "quantile_tf", "total_variation_tf", "dfa", "pfd", "hfd", "kfd",
        "lziv_complexity", "shannon_entropy", "permutation_entropy", "spectral_entropy", "svd_entropy", "app_entropy",
        "sample_entropy")
    header_len = (len(headers_array)-1)
    headers = ",".join(headers_array)

    # Exporting Features as CSV File
    all_features = (
        seizure_check, peak_amplitude, peak_to_peak_amplitude, rms, line_length, quantile, min_amplitude, zcr, energy,
        variance, std_dvt, median, mean, hjort_acivity, hjort_mobility, hjort_complexity, skewness, kurtosis,
        fano_factor, form_factor, pulse_indicator, total_variation, band_power_delta, band_power_theta,
        band_power_alpha, band_power_beta, line_length_delta, line_length_theta, line_length_alpha, line_length_beta,
        quantile_delta, quantile_theta, quantile_alpha, quantile_beta, mean_delta, mean_theta, mean_alpha, mean_beta,
        median_delta, median_theta, median_alpha, median_beta, std_dvt_delta, std_dvt_theta, std_dvt_alpha,
        std_dvt_beta, var_delta, var_theta, var_alpha, var_beta, cA, cD, mean_tf, quantile_tf, total_variation_tf, dfa,
        pfd, hfd, kfd, lziv_complexity, shannon_entropy, permutation_entropy, spectral_entropy, svd_entropy,
        app_entropy, sample_entropy)
    combined = np.column_stack(all_features)
    np.savetxt(save_path + edf_files[i].replace('.edf', '.csv'), combined, delimiter=",", header=headers,
               fmt=','.join(['%d'] + ['%1.5f'] * header_len), comments="")