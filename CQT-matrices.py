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

import pandas as pd
import glob

path =  # use your path

all_files = glob.glob(path + "*/*/*.wav")
all_files = sorted(all_files)
file_names = [i.split('/')[-1] for i in all_files]

from collections import Iterable

def transform(all_files):
    tmp_ftr = []
    file_names = [i.split('/')[-1] for i in all_files]
    feature = []
    min_size = 1000000
    for i in np.arange(len(all_files)):
#         print(i)
        y, sr = librosa.load(all_files[i])
        trans = librosa.feature.chroma_cqt(y, sr=sr)
        trans = trans.T
#         size = trans.shape[1]
#         print(i, ": ", trans.shape)
        flat = trans.flatten()
        size =len(flat)
        if min_size > size:
            min_size = size
        feature.append([file_names[i], *list(flat)])
#         feature.append(trans.flatten())
        
    return feature, min_size

chunk = 1000
# min_size = 700

for i in range(12, 16):
    down = i * chunk
    up = (i+1) * chunk
    try:
        subset = all_files[down: up]
    except IndexError:
        subset = all_files[down:]
#     print("up: ", up, " ; down: ", down)
    # tmp_ftr = getDF
    start = time.time()
    DF_features, size = transform(subset)
#     if min_size > size:
#         min_size = size
    df = pd.DataFrame(DF_features)
    end = time.time()
    delta = end - start
    print("Take time: ", delta)
    df.to_csv('./OutputData/Transform_'+ str(i) + '.csv')
    print("min_size: ", size)    
    # break
    
# print("min_size: ", min_size)