import os 
import librosa
from glob import glob
import librosa.display 
import matplotlib.pyplot as plt
import random
import IPython.display as ipd
import fnmatch
import itertools
import numpy as np
from types import *
import pandas as pd
import librosa as lr
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import glob

import pandas as pd
import glob

path = r'/Users/kirsten/Documents/USF/Intersession/Project/sid_dev2' # use your path
all_files = glob.glob(path + "*/*/*.wav")
file_names = [i.split('/')[-1] for i in all_files]

from collections import Iterable
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

feature_cols = ['FileName','Centroid', 'variance','skewness','kurtosis',['mfcc'+str(i) for i in np.arange(12)+1], 
                'roll_off_max', 'roll_off_min', 'flatness', 'zeroCrossingRate', 'rms', ['cqt'+str(i) for i in np.arange(12)]]
feature_cols = list(flatten(feature_cols))
#feature_cols

def getDF(all_files):
    tmp_ftr = []
    file_names = [i.split('/')[-1] for i in all_files]
    #freq_range = [0, 1000]

    for i in np.arange(len(all_files)): 
        y, sr = librosa.load(all_files[i])

        #features calculated over all time, thus taking averages over time interval (all wav file here)
        cntrd = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(cntrd) 
        #centroid_std = np.std(cntrd)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfccs_mean = np.mean(mfccs, axis = 1)
        
        cqt = librosa.feature.chroma_cqt(y, sr=sr)
        cqt_mean = np.mean(cqt, axis = 1)

        bandwidth_2 =librosa.feature.spectral_bandwidth(y=y, sr=sr, p = 2)
        bandwidth_3 =librosa.feature.spectral_bandwidth(y=y, sr=sr, p = 3)
        bandwidth_4 =librosa.feature.spectral_bandwidth(y=y, sr=sr, p = 4)

        #sp_ft = librosa.stft(y)
        #sp_db = librosa.amplitude_to_db(abs(sp_ft))
        #ln = np.mean(sp_db[freq[0]:freq[1], :], axis = 1) # lineout selec freq over all time
        #freqs = freq_range[0]+np.arange(len(ln))*(freq_range[1]-freq_range[0])/(len(ln) -1)
        #max_f = freqs[np.where(ln == np.max(ln))[0][0]]

        roll_off_max = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent = 0.8)
        roll_off_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent = 0.1)
        flatness = librosa.feature.spectral_flatness(y=y)
        zeroCross = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)

        #max_f
        feature_ls = [file_names[i],centroid_mean, np.mean(bandwidth_2), np.mean(bandwidth_3), 
                      np.mean(bandwidth_4), mfccs_mean, np.mean(roll_off_max), 
                      np.mean(roll_off_min), np.mean(flatness), np.mean(zeroCross),
                     np.mean(rms), cqt_mean]
        feature_ls = list(flatten(feature_ls))


        tmp_ftr.append(feature_ls)
        
            
    DF = pd.DataFrame(tmp_ftr, columns = feature_cols )
    return DF

chunk = 1000

for i in range(16):
    down = i * chunk
    up = (i+1) * chunk
    try:
        subset = all_files[down: up]
    except IndexError:
        subset = all_files[down:]
#     print("up: ", up, " ; down: ", down)
    # tmp_ftr = getDF
    start = time.time()
    DF_features = getDF(subset)
    end = time.time()
    delta = end - start
    print("Take time: ", delta)
    DF_features.to_csv('./CQT_2/'+ str(i) + '.csv')
#     break