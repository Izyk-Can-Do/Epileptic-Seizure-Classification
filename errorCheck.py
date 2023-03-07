import os
import numpy as np
import pyedflib as edf

chb = 'chb15/'
dir_path = os.path.dirname(os.path.realpath(__file__))
read_path = dir_path + '/EDF Files/' + chb
edf_files = sorted([file for file in os.listdir(read_path) if file.endswith('.edf')])
print('List of edf files: %s' % edf_files)

for i in range(len(edf_files)):
    f = edf.EdfReader(read_path + edf_files[i])
    duration = int(f.getFileDuration())
    signal_full = f.readSignal(20)
    signal_divided = np.split(signal_full, duration)
    check_nan = sum(map(lambda row: np.max(row) == np.min(row), signal_divided))
    if check_nan != 0:
        print('Measurement errors in the %s' % edf_files[i])
