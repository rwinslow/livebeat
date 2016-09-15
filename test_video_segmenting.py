import os
import re
import sys

from PIL import Image, ImageFilter
import imageio
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from scipy import ndimage as ndi
from scipy.misc import imresize
from skimage import feature
from skimage.measure import block_reduce
from skimage import color
from scipy.signal import convolve2d as conv2





def file_list(start_dir):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list

def rgb2vec(img):
    return [img[:,:,i].flatten().tolist() for i in range(0, 3)][0]

def gray2vec(img):
    return img.flatten().tolist()

def img2val(img):
    return [sum(img[:,:,i].flatten().tolist()) for i in range(0, 3)]

def edge_detect_score(img):
    score = [np.sum(img, axis=1) for img in [feature.canny(img[:,:,i]) for i in range(3)]]
    score = [item for sublist in score for item in sublist]
    return score

def edge_detect_gray_score(img):
    return np.sum(feature.canny(grayscale_image(img)), axis=1)

def score_channel_cols(img):
    score = [sum(np.sum(img[:,:,i], axis=1)) for i in range(3)]
    return score

def downsample_image(img):
    return block_reduce(img, block_size=(2, 2, 1), func=np.mean)

def grayscale_image(img):
    return color.rgb2gray(img)

def scale_features(img):
    mean = np.mean(img)
    return []

def img2score(img):
    score = []
    # score.extend(score_channel_cols(img))
    # score.extend(edge_detect_score(img))
    score.extend(edge_detect_gray_score(img))
    # score = gray2vec(grayscale_image(downsample_image(img)))
    return score

def train_scene_detection():
    """Train scene detection classifier"""

    print('Training scene detection classifier')

    X = []
    y = []

    positive_list = file_list('./train_game')
    negative_list = file_list('./train_non-game')

    for f in positive_list:
        im = np.array(Image.open(os.path.join('./train_game', f)))
        X.append(img2score(im))
        y.append(1)

    for f in negative_list:
        im = np.array(Image.open(os.path.join('./train_non-game', f)))
        X.append(img2score(im))
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


def segment(filename, model, codec='ffmpeg', start_pct=0, end_pct=1):
    """Generate timecodes for game and non-game segments"""
    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, codec)

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    nframes = meta['nframes']
    frames = np.arange(int(nframes*start_pct), int(nframes*end_pct), fps*10)

    # Check frames
    timecodes = []
    start_time = 0
    end_time = 0
    for i in frames:
        img = vid.get_data(i)

        # Resize image as necessary
        img = imresize(img, (720, 1280))
        # imageio.imwrite('./test_images_old_vids/frame_old_{}.png'.format(i), img)

        # Shop button
        h, w, c = img.shape
        x1 = int(w * .94) + np.random.random_integers(-2, 1)
        # x1 = w - 116
        x2 = x1 + 76
        y1 = int(h * .814) + np.random.random_integers(-2, 2)
        # y1 = h - 201
        y2 = y1 + 26
        # print(w, x1, x2)
        # print(h, y1, y2)
        shop = img[y1:y2, x1:x2, :]
        imageio.imwrite('./test_images_old_vids/shop_new_{}.png'.format(i), shop)

        threshold = 0.5
        prediction = model.predict(np.array(img2score(shop)).reshape(1, -1))
        if prediction >= threshold:
            if not start_time:
                start_time = '{}'.format(i)
                # imageio.imwrite('./test_images/shop_start_{}.png'.format(i), shop)
                # imageio.imwrite('./test_images/full_start_{}.png'.format(i), img)
            if prediction <= threshold and start_time:
                end_time = '{}'.format(i)
                timecodes.append(','.join([start_time, end_time]))
                print([start_time, end_time])
                start_time = 0
                end_time = 0
                # imageio.imwrite('./test_images/shop_end_{}.png'.format(i), shop)
                # imageio.imwrite('./test_images/full_end_{}.png'.format(i), img)

    vid.close()

    return timecodes

# Process arguments from command line
try:
    start_pct = float(sys.argv[1])
    end_pct = float(sys.argv[2])
except:
    print('No range arguments. Processing 100% of video.')
    start_pct = 0
    end_pct = 1

# filename = './video/dota2ti_v83196893_720p30.mp4'
filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v29880976_720p30.mp4'
timecodes = segment(filename, train_scene_detection(), 'ffmpeg', start_pct, end_pct)
print('Done. Now writing timecodes.')

# Get video id
video_id = re.findall('v\d+', filename)[0]

timecode_file = './timecodes/timecodes_{}_s{}_e{}.csv'.format(video_id, start_pct, end_pct)
with open(timecode_file, 'w') as f:
    for v in timecodes:
        f.write(v)
        f.write('\n')
