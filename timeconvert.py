import datetime

import numpy as np
import pandas as pd

def to_timecode(t):
    t = int(t)
    h = int(t / 3600)
    m = int((t % 3600) / 60)
    s = int((t % 3600) % 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)

df = pd.read_csv('test.csv', header=None, names=['time'], dtype=str)
df['time'] = df['time'].apply(to_timecode)

for i, v in enumerate(df['time'].values):
    ffmpeg = 'ffmpeg -ss {} -i dota2ti_v83196893_720p30.mp4 -vframes 1 out_{}.png'.format(v, i)
    os.system(ffmpeg)

# with open('codes.txt', 'w') as f:
#     for v in df['time'].values:
#         f.write(v)
#         f.write('\n')
