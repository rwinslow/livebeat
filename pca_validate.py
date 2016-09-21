import os
import re
import sys

from PIL import Image, ImageFilter
import imageio
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from scipy import ndimage as ndi
from scipy.misc import imresize
from skimage import feature
from skimage.measure import block_reduce
from skimage import color
from scipy.signal import convolve2d as conv2
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn import decomposition
import seaborn as sns
sns.set(style="ticks", color_codes=True)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    cols = list(np.sum(feature.canny(grayscale_image(img), sigma=1), axis=1))
    # rows = list(np.sum(feature.canny(grayscale_image(img), , sigma=1), axis=0))
    # cols.extend(rows)
    return cols

def corner_find_features(img):
    return np.sum(feature.corner_fast(img[:,:,0], threshold=.05), axis=1)

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

def hog_img(img):
    img = np.asarray(img)
    hog_img = grayscale_image(img)
    hogged, hogged_img = feature.hog(hog_img, visualise=True)
    return hogged

def img2score(img):
    score = []
    score = hog_img(img)
    return score

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


def pca_check():
    """Train scene detection classifier"""

    print('Training scene detection classifier')

    X = []
    y = []

    positive_list = file_list('./test_images_button')
    negative_list = file_list('./test_images_non-button')

    for f in positive_list:
        im = np.array(Image.open(os.path.join('./test_images_button', f)))
        X.append(img2score(im))
        y.append(1)

    for f in negative_list:
        im = np.array(Image.open(os.path.join('./test_images_non-button', f)))
        X.append(img2score(im))
        y.append(0)

    # X = normalize(X)

    # PCA
    # pca = decomposition.PCA(n_components=4)
    # pca.fit(X)
    # X = pca.transform(X)

    # Train logistic model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    pca_train = decomposition.PCA(n_components=4)
    pca_train.fit(X_train)
    X_train = pca_train.transform(X_train)

    pca_test = decomposition.PCA(n_components=4)
    pca_test.fit(X_test)
    X_test = pca_test.transform(X_test)

    # Generate pairs Plot
    print('Plotting train pairs plot')
    X_train = pd.DataFrame(X_train)
    X_train['label'] = pd.DataFrame(
            ['Game' if v == 1 else 'Non Game' for v in y_train]
    )
    X_train.columns=['PC1', 'PC2', 'PC3', 'PC4', 'label']
    g = sns.pairplot(X_train, hue='label')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.savefig('/Users/Rich/Documents/Twitch/pca-result/pca-train.png')

    print('Plotting test pairs plot')
    X_test = pd.DataFrame(X_test)
    X_test['label'] = pd.DataFrame(
            ['Game' if v == 1 else 'Non Game' for v in y_test]
    )
    X_test.columns=['PC1', 'PC2', 'PC3', 'PC4', 'label']
    g = sns.pairplot(X_test, hue='label')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.savefig('/Users/Rich/Documents/Twitch/pca-result/pca-test.png')

    return True


filename = './video/dota2ti_v83196893_720p30.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v29880976_720p30.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v83196893_720p30_game.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v83012529_720p30_nongame_2.mp4'

pca = pca_check()
