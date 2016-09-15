import csv
import json
import os
import re
import sys

from PIL import Image, ImageFilter
import imageio
import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy import ndimage as ndi
from skimage import feature
from scipy.fftpack import rfft, irfft, fftfreq, rfftfreq
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from skimage.measure import block_reduce

# For closing ffmpeg
import subprocess, signal
p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
out, err = p.communicate()

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

def pull_frame(filename, seconds, chat_val):
    """Pull frame at n seconds"""
    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get frame
    img = vid.get_data(i * 60)

    # save file with chat frequency for training
    basedir = '/Volumes/Passport/LiveBeat/video/v82878048_training'
    imageio.imwrite(os.path.join(basedir, 'frame_{}_s{}_c{}.png'.format(video_id, seconds, chat_val)), img)



    # Close video connection
    vid.close()

    for line in out.splitlines():
        line = line.decode('utf-8')
        if 'ffmpeg.osx' in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)


    return img

def edge_detect_score(img):
    score = [np.sum(img, axis=1) for img in [feature.canny(img[:,:,i]) for i in range(3)]]
    score = [item for sublist in score for item in sublist]
    return score

def downsample_image(img):
    return block_reduce(img, block_size=(4, 4, 1), func=np.mean)

def generate_score(img):
    img = downsample_image(img)
    return edge_detect_score(img)


# Get file list
filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'
# filename = './video/dota2ti_v82878048_360p30.avi'
start = '/Volumes/Passport/LiveBeat/chat/v82878048'
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

# Create data frame from chat data and convert ms to s
df = pd.DataFrame(df)
minimum = df['timestamp'].min()
maximum = df['timestamp'].max()
df['timestamp'] = df['timestamp'].apply(lambda x: x - minimum)
df['secondstamp'] = df['timestamp'].apply(lambda x: int(round(x/1000)))

# Create chat frequency data frame where index is no. of seconds into video
chat_freq = df['secondstamp'].value_counts().sort_index()
chat_freq = pd.DataFrame(chat_freq)
chat_freq.columns = ['frequency']
cf_copy = chat_freq.copy()

# Get video id
video_id = re.findall('v\d+', filename)[0]

# Open file handle
vid = imageio.get_reader(filename, 'ffmpeg')

# Get metadata
meta = vid.get_meta_data()
fps = int(meta['fps'])
nframes = meta['nframes']
frames = np.arange(0, int(nframes), 1)

# Close video
vid.close()

# Convert frames to seconds and initialize data frame
values = [0] * int(len(frames)/fps)
df = pd.DataFrame(values).reset_index()
df.columns = ['second', 'game']
smear_df = df.copy()

# Import timecode data
timecodes = pd.read_csv('./timecodes/timecodes_v82878048.csv', names=['start', 'stop'])
timecodes = timecodes.astype(int)

# Smear timecodes to remove anything < 30 seconds long
smear_tc = timecodes[timecodes['stop'] - timecodes['start'] >= 30].reset_index(drop=True)

# Run through smeared timecodes and flip switches on correct seconds of video
for i in smear_tc.index:
    row = smear_tc.iloc[i][['start', 'stop']]
    start = row[0]/fps
    stop = row[1]/fps
    df.loc[(df.index >= start) & (df.index < stop), 'game'] = 1

df_copy = df[0:len(cf_copy)]
cf_copy['game'] = df_copy['game']
cf_copy.loc[cf_copy['game'] != 1] = 0

# cf_copy containes chat frequency, game/not game, and index = seconds
times = list(cf_copy[cf_copy['game'] == 1].index)
chat_freq_values = list(cf_copy[cf_copy['game'] == 1]['frequency'])

X = []
y = []
i = 2000
while i < int(len(times)):
    try:
        seconds = times[i]
        second_increment = 10
        chats = sum(chat_freq_values[i:i+second_increment])
        img = pull_frame(filename, seconds, chats)
        y.append(chats)
        X.append(generate_score(img))
        i += second_increment
    except:
        print('Last i:', i)
        print('Length:', len(times))
        raise
