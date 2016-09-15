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
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model, datasets, metrics
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from skimage import feature


def file_list(start_dir):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list

def img2vec(img):
    return [img[:,:,i].flatten().tolist() for i in range(0, 3)][0]

def img2val(img):
    return [sum(img[:,:,i].flatten().tolist()) for i in range(0, 3)]

def img2edgesum(img):
    # return [sum(row) for row in feature.canny(img[:,:,0])]
    score = []
    score = [sum(sum(img)) for img in [feature.canny(img[:,:,i]) for i in range(3)]]
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
        X.append(img2vec(im))
        y.append(1)

    for f in negative_list:
        im = np.array(Image.open(os.path.join('./train_non-game', f)))
        X.append(img2vec(im))
        y.append(0)

    # Train logistic model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression().fit(X_train, y_train)

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, y_train)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, y_train)

    ###############################################################################
    # Evaluation

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            y_test,
            classifier.predict(X_test))))

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            y_test,
            logistic_classifier.predict(X_test))))

    return classifier


def segment(filename, model, codec='ffmpeg', start_pct=0, end_pct=0.25):
    """Generate timecodes for game and non-game segments"""
    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, codec)

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    nframes = meta['nframes']
    frames = np.arange(int(nframes*float(start_pct)), int(nframes*float(end_pct)), fps*30)

    # Check frames
    timecodes = []
    start_time = 0
    end_time = 0
    for i in frames:
        if not i % (fps * 3600):
            print('Hour')

        try:
            img = vid.get_data(i)
        except:
            return timecodes

        # Shop button
        h, w, c = img.shape
        x1 = int(w * .945)
        x2 = x1 + 64
        y1 = int(h * .814)
        y2 = y1 + 22
        shop = img[y1:y2, x1:x2, :]

        threshold = 0.5
        prediction = model.predict(np.array(img2vec(shop)).reshape(1, -1))
        if prediction >= threshold:
            if not start_time:
                start_time = '{}'.format(i)
        if prediction <= threshold and start_time:
                end_time = '{}'.format(i)
                timecodes.append(','.join([start_time, end_time]))
                start_time = 0
                end_time = 0

    vid.close()

    return timecodes

# Process arguments from command line
try:
    start_pct = sys.argv[1]
    end_pct = sys.argv[2]
except:
    print('No range arguments. Processing first 25% of video.')
    start_pct = 0
    end_pct = 0.25

filename = './video/dota2ti_v83196893_720p30.mp4'
timecodes = segment(filename, train_scene_detection(), 'ffmpeg', start_pct, end_pct)
print('Done. Now writing timecodes.')

timecode_file = 'timecodes_s{}_e{}.csv'.format(float(start_pct)*100, float(end_pct)*100)
with open(timecode_file, 'w') as f:
    for v in timecodes:
        f.write(v)
        f.write('\n')
