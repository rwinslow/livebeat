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

from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score
from skimage import feature, color, exposure
from skimage.measure import block_reduce
from sklearn.learning_curve import learning_curve, validation_curve

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

def pull_frame(filename, seconds, fps=60):
    """Pull frame at n seconds"""
    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get frame
    img = vid.get_data(i * fps)

    # Close video connection
    vid.close()

    return img

def get_player_status(img):
    # Player status
    h, w, c = img.shape
    factor = 0.23
    y1 = int(h * 0.04)
    x1 = int(w * (0.5 - factor))
    y2 = int(h * 0.0645)
    x2 = int(w * (0.5 + factor))
    pl_status = crop_rect(img, x1, y1, x2, y2)

    # Block out center of player status
    h, w, c = pl_status.shape
    factor = .12
    x1 = int(w * (0.5 - factor))
    x2 = int(w * (0.5 + factor))
    pl_status = blackout_middle(pl_status, x1, x2)

    return pl_status

def edge_detect_score(img):
    score = [np.sum(img, axis=1) for img in [feature.canny(img[:,:,i]) for i in range(3)]]
    score = [item for sublist in score for item in sublist]
    return score

def downsample_image(img):
    return block_reduce(img, block_size=(2, 2, 1), func=np.mean)

def crop_rect(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2, :]

def blackout_middle(img, x1, x2):
    img[:, x1:x2, :] = 0
    return img

def hist_sum_log_hsv(img):
    img = Image.fromarray(img, 'RGB')
    hsv = img.convert('HSV')
    hist = hsv.histogram()
    return [np.log(sum(hist))]

def hist_chan_sum_rgb(img):
    red = Image.fromarray(img[:,:,0])
    green = Image.fromarray(img[:,:,1])
    blue = Image.fromarray(img[:,:,2])

    red = np.log(sum(red.histogram()))
    green = np.log(sum(green.histogram()))
    blue = np.log(sum(blue.histogram()))
    return (red, green, blue)

def rgb2vec(img):
    return [img[:,:,i].flatten().tolist() for i in range(0, 3)][0]

def gray2vec(img):
    return img.flatten().tolist()

def img2val(img):
    return [sum(img[:,:,i].flatten().tolist()) for i in range(0, 3)]

def img2row(img):
    score = [np.sum(mat, axis=1) for mat in img]
    score = [item for sublist in score for item in sublist]
    return score

def img2hog(img):
    img = Image.fromarray(img, 'RGB').convert('L')
    fd, hog_image = feature.hog(img, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualise=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


def logistic_features(img):
    features = []
    # score.extend(score_channel_cols(img))
    features.extend(edge_detect_score(img))
    # score = gray2vec(grayscale_image(downsample_image(img)))
    return features

def linear_features(img):
    features = []
    img = get_player_status(img)
    # img = downsample_image(img)
    # score = edge_detect_score(img)
    # img2hog(img)
    features.extend(hist_sum_log_hsv(img))
    features.extend(hist_chan_sum_rgb(img))
    return features

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def train_scene_detection():
    """Train scene detection classifier"""

    print('Training scene detection classifier')

    X = []
    y = []

    positive_list = file_tree('./train_game')
    negative_list = file_tree('./train_non-game')

    for f in positive_list:
        im = np.array(Image.open(os.path.join('./train_game', f)))
        X.append(logistic_features(im))
        y.append(1)

    for f in negative_list:
        im = np.array(Image.open(os.path.join('./train_non-game', f)))
        X.append(logistic_features(im))
        y.append(0)

    # Train logistic model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression().fit(X_train, y_train)

    predicted = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Generate evaluation metrics
    print('Accuracy:', metrics.accuracy_score(y_test, predicted))
    print(metrics.roc_auc_score(y_test, probs[:, 1]))
    print('Confusion Matrix:', metrics.confusion_matrix(y_test, predicted))
    print('Classification Report:', metrics.classification_report(y_test, predicted))

    # evaluate the model using 10-fold cross-validation
    # scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    # print('10-fold Cross-Validation:', scores.mean())

    return model


def segment(filename, model, seconds_between_frame_grabs=30):
    """Generate timecodes for game and non-game segments"""
    print('Getting "game" timecodes')

    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    nframes = meta['nframes']
    frames = np.arange(0, nframes, fps*seconds_between_frame_grabs)

    # Check frames
    timecodes = []
    start_time = 0
    end_time = 0
    for i in frames:
        img = vid.get_data(i)

        # Shop button
        h, w, c = img.shape
        x1 = int(w * .94)
        x2 = x1 + 76
        y1 = int(h * .814)
        y2 = y1 + 26
        shop = img[y1:y2, x1:x2, :]

        threshold = 0.5
        prediction = model.predict(np.array(logistic_features(shop)).reshape(1, -1))
        if prediction >= threshold:
            if not start_time:
                start_time = '{}'.format(i)
            if prediction <= threshold and start_time:
                end_time = '{}'.format(i)
                timecodes.append([start_time, end_time])
                print('Found game:',[start_time, end_time])
                start_time = 0
                end_time = 0

    vid.close()
    return timecodes

def chat2df(path):
    file_list = file_tree(path)
    chats = []

    # Aggregate chat files into dictionary
    for fname in file_list:
        get_path = os.path.join(start, fname)
        with open(get_path) as f:
            # Format line and separate multiple JSON strings with commas
            line = '[{}]'.format(f.readline()).replace('}}{', '}},{')
            data = json.loads(line)[0]

            for message in data['data']:
                chats.append(process_data(message))

    # Create data frame from chat data and convert ms to s
    chats = pd.DataFrame(chats)
    minimum = chats['timestamp'].min()
    maximum = chats['timestamp'].max()
    chats['timestamp'] = chats['timestamp'].apply(lambda x: x - minimum)
    chats['secondstamp'] = chats['timestamp'].apply(lambda x: int(round(x/1000)))

    return chats

def get_video_data(path):
    # Get video id
    video_id = re.findall('v\d+', path)[0]

    # Open file handle
    vid = imageio.get_reader(path, 'ffmpeg')

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    nframes = meta['nframes']
    frames = np.arange(0, int(nframes), 1)

    # Close video
    vid.close()

    return (fps, frames)

# Get file list
filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'
start = '/Volumes/Passport/LiveBeat/chat/v82878048'
file_list = file_tree(start)

# Create chat frequency data frame where index is no. of seconds into video
df = chat2df(start)
chat_freq = pd.DataFrame(df['secondstamp'].value_counts().sort_index())
chat_freq.columns = ['frequency']
cf_copy = chat_freq.copy()

# Get video id
fps, frames = get_video_data(filename)

# Convert frames to seconds and initialize data frame
values = [0] * int(len(frames)/fps)
df = pd.DataFrame(values).reset_index()
df.columns = ['second', 'game']
smear_df = df.copy()

# Get timecode data
# timecodes = pd.read_csv('./timecodes/timecodes_v82878048.csv', names=['start', 'stop'])
timecodes = pd.DataFrame(
    [[57600, 120600],
    [205200, 334800],
    [450000, 451800],
    [453600, 574200],
    [657000, 790200],
    [927000, 1200600],
    [1297800, 1436400]]
)

# timecodes = pd.DataFrame(segment(filename, train_scene_detection()))
timecodes = timecodes.astype(int)
timecodes.columns = ['start', 'stop']

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
i = 0
print('Reading frames')
times_length = len(times)
increments = [int(i*times_length) for i in np.arange(0, 1, .1)]
i = 0
while i < times_length:
    if i in increments:
        i += 10
        print(int(i/times_length), '% done reading frames')

    seconds = times[i]
    second_increment = 1
    img = pull_frame(filename, seconds)
    chats = sum(chat_freq_values[i:i+second_increment])
    chats = np.log(chats)

    y.append(chats)
    X.append(linear_features(img))
    i += second_increment
print('100% done reading frames')

print('Saving X, y')
vectors = pd.DataFrame([X, y]).to_csv(filename + '_features.csv')

print('Training model')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

print('Plotting learning curve')
plot_learning_curve(model, 'learning curve', X_train, y_train)
plt.show()

print('Cross validation 10-fold and plot')
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(model, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
