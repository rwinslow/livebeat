import os
import re
import sys
import warnings

from PIL import Image, ImageFilter
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.linear_model import LogisticRegression
from scipy.misc import imresize
from skimage import feature, color

# Hide scikit-learn warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def file_list(start_dir):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list

def grayscale_image(img):
    return color.rgb2gray(img)

def hog_img(img):
    # img = np.asarray(img)
    hog_img = grayscale_image(img)
    hogged, hogged_img = feature.hog(hog_img, visualise=True)
    return hogged

def img2features(img):
    return hog_img(img)

def scene_detection():
    """Train scene detection classifier"""

    # Print status
    print('Training scene detection classifier')

    # Build training set of classified images
    positive_list = file_list('./test_images_button')
    negative_list = file_list('./test_images_non-button')
    X = []
    y = []

    for f in positive_list:
        img = np.array(Image.open(os.path.join('./test_images_button', f)))
        X.append(img2features(img))
        y.append(1)

    for f in negative_list:
        img = np.array(Image.open(os.path.join('./test_images_non-button', f)))
        X.append(img2features(img))
        y.append(0)

    # PCA to get valuable features
    pca = decomposition.PCA(n_components=4)
    pca.fit(X)
    X = pca.transform(X)

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    return (model, pca)


def segmenter(filename, model, pca, threshold=0.5, seconds_between_frames=30):
    """Generate timecodes for game and non-game segments"""

    # Print status
    print('Finding timecodes for segments')

    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get metadata and select 1 frame every n seconds
    meta = vid.get_meta_data()
    fps = int(np.round(meta['fps']))
    nframes = meta['nframes']
    frames = np.arange(0, nframes, seconds_between_frames*fps)

    # Check frames
    timecodes = []
    start_time = end_time = 0

    for i in frames:
        img = vid.get_data(i)

        # Resize image appropriately for training set
        img = imresize(img, (720, 1280))

        # Isolate shop button
        h, w, c = img.shape
        x1 = int(w * .94)
        x2 = x1 + 76
        y1 = int(h * .814)
        y2 = y1 + 26
        shop = img[y1:y2, x1:x2, :]

        # Generate predictions for each selected frame
        features = pca.transform(img2features(shop))
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        if prediction >= threshold:
            if not start_time:
                start_time = '{}'.format(i)
        if prediction <= threshold and start_time:
            end_time = '{}'.format(i)
            timecodes.append(','.join([start_time, end_time]))
            start_time = 0
            end_time = 0

    # Close video handle to release thread and buffer
    vid.close()

    return timecodes

filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'

model, pca = scene_detection()
timecodes = segmenter(filename, model, pca)
