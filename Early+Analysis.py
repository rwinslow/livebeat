
# coding: utf-8

# In[1]:

import json
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, rfftfreq


# In[23]:

start = './chat/'

def file_tree(start):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list


def process_data(data):
    """Process JSON data"""
    attr = data['attributes']

    timestamp = attr['timestamp']
    message = attr['message']
    author = attr['from']
    turbo = attr['tags']['turbo']
    sub = attr['tags']['subscriber']

    try:
        emotes = attr['tags']['emotes']
        emote_count = sum([len(emotes[key]) for key in emotes.keys()])
    except:
        emote_count = 0

    row = {
        'timestamp': timestamp,
        'author': author,
        'message': message,
        'turbo': turbo,
        'sub': sub,
        'emote_count': emote_count
    }

    return row


def high_pass_filter(freq, amp, min_freq):
    df = pd.DataFrame(amp, index=freq, columns=['Amplitude'])
    df[df.index < min_freq] = 0
    return df

def round_to_val(x, val=30):
    return int(np.round(x/val) * val)


# In[3]:

# Get file list
file_list = file_tree(start)
df = []

# Aggregate files into dictionary
for file in file_list:
    get_path = os.path.join(start, file)
    with open(get_path) as f:
        # Format line and separate multiple JSON strings with commas
        line = '[{}]'.format(f.readline()).replace('}}{', '}},{')
        data = json.loads(line)[0]

        for message in data['data']:
            df.append(process_data(message))

# Create data frame from chat data
df = pd.DataFrame(df)


# In[4]:

minimum = df['timestamp'].min()
maximum = df['timestamp'].max()
df['timestamp'] = df['timestamp'].apply(lambda x: x - minimum)
df['secondstamp'] = df['timestamp'].apply(lambda x: int(round(x/1000)))


# In[5]:

df.iloc[range(100,106)]


# In[6]:

counts = df['secondstamp'].value_counts().sort_index()


# In[7]:

x = counts.index.values
y = counts.values


# In[8]:

# fig, ax = plt.subplots(figsize=(16, 6))
# ax.scatter(x, y)
# plt.show()


# In[9]:

# Generate CDF
# cdf = counts.sort_values()
# cum_dist = np.linspace(0.,1.,len(cdf))
# cdf = pd.Series(cum_dist, index=cdf)
# cdf.plot(drawstyle='steps')


# In[10]:

# fig = plt.figure(figsize=(19,2))
# ax = df['secondstamp'].hist(bins=len(file_list))
# ax.set_xlabel('Progress')
# ax.set_ylabel('Count')


# In[11]:

time = x
signal = y

W = rfftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)


# In[12]:

# plt.subplot(211)
# plt.plot(time,signal)
# plt.subplot(212)
# plt.plot(W,f_signal)
# plt.show()


# In[26]:

filtered_rfft = high_pass_filter(W, f_signal, 0.3)
plt.plot(filtered_rfft.index, filtered_rfft['Amplitude'])


# In[ ]:
filtered = irfft(filtered_fft['Amplitude'], len(filtered_fft['Amplitude']))


# In[27]:

# print(filtered)


# In[ ]:
