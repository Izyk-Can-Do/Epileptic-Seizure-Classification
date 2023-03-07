import pandas as pd
import scipy.stats as sstts

def shn_entropy(array):
    pd_series = pd.Series(array)
    counts = pd_series.value_counts()
    shn_entropy = sstts.entropy(counts)
    return shn_entropy