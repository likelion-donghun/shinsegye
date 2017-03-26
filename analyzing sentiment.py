import pyprind
import pandas as pd
import os

pbar = pyprind.ProgBar(50000)

labels={'pos':1, 'neg':0}

df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' %(s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='UTF8', errors='ignore') as infile: #, encoding='UTF8'(http://airpage.org/xe/language_data/20205), errors='ignore'(http://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c) 추가
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
