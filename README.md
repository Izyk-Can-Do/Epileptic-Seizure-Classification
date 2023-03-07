# Epileptic Seizure Detection

This project was developed based on the CHB-MIT Scalp EEG Database.

**Note: Needed File Formats** *EDF Files/chbxx/chbxx_xx.edf, CSV Files/chbxx/chbxx_xx.csv*

## Feature Extraction

A total of 64 features were extracted.

**Time Domain Features**
- RMS
- Fano factor
- Kurtosis
- Skewness
- Hjorth complexity
- Hjorth mobility
- Hjorth activity
- Mean
- Median
- Standard deviation
- Variance
- Energy
- Zero crossing rate
- Peak amplitude
- Peak to peak amplitude
- Min amplitude
- Total variation
- Line length
- Quantile


**Frequency Domain Features**
- Band powers
- Standard deviation
- Variance
- Median
- Mean
- Quantile
- Line length
- Wavelet coefficients

**Time-Frequency Domain Features**
- Mean
- Quantile
- Total variation

**Non-Linear Features**
- Lempel-Ziv complexity
- Detrended fluctuation
- Katz fractal dimension
- Petrosian fractal dimension
- Higuchi fractal dimension


**Entropy Features**
- Shannon entropy
- Spectral entropy
- Singular value decomposition entropy
- Approximate entropy
- Sample entropy
- Permutation entropy
