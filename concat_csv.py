import pandas as pd
import glob

path = 'new_chunk/'
all_files = glob.glob(path + "*/*.csv")
big_csv = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    big_csv.append(df)

frame = pd.concat(big_csv, axis=0, ignore_index=True)

frame.drop(['Unnamed: 0'],axis=1,inplace=True)
frame.to_csv('final_chunk.csv', index=False)