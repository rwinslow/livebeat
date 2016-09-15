import os
import re

import imageio
import numpy as np
from PIL import Image, ImageFilter


def crop_rect(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2, :]

def blackout_middle(img, x1, x2):
    img[:, x1:x2, :] = 0
    return img

def crop_shop(img, i, count):
    # Shop button
    h, w, c = img.shape
    x1 = int(w * .945)
    x2 = int(w * .996)
    y1 = int(h * .814)
    y2 = int(h * .847)
    shop = img[y1:y2, x1:x2, :]
    name_shop = 'shop_fr{}_count{}.png'.format(i, count)
    path_shop = os.path.join(basedir, 'shop', name_shop)
    imageio.imwrite(path_shop, shop)

    return True

def crop_pl_status(img, i, count):
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
    name_pl_status = 'pl_status_fr{}_count{}.png'.format(i, count)
    path_pl_status = os.path.join(basedir, 'pl_status', name_pl_status)
    imageio.imwrite(path_pl_status, pl_status)

    return True

def crop_map(img, i, count):
    # Map
    h, w, c = img.shape
    x1 = 0
    y1 = int(h * .736)
    x2 = int(w * .153)
    y2 = int(h * .986)
    map_roi = crop_rect(img, x1, y1, x2, y2)
    name_map_roi = 'map_roi_fr{}.png'.format(i)
    path_map_roi = os.path.join(basedir, 'map_roi', name_map_roi)
    imageio.imwrite(path_map_roi, map_roi)

    return True

def prep(filename, codec='ffmpeg', multiplier=1, t0=0, count=0):
    """Cut video to 1 frame per second and save images"""

    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, codec)

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    frames = np.arange(t0, meta['nframes'], fps * multiplier)

    # Make directories for storage
    basedir = '/Volumes/Passport/LiveBeat/video/{}_1fp{}s'.format(video_id, multiplier)
    try:
        os.makedirs(os.path.join(basedir, 'pl_status'))
    except FileExistsError:
        pass

    try:
        os.makedirs(os.path.join(basedir, 'shop'))
    except FileExistsError:
        pass

    return frames

filename = './video/dota2ti_v83196893_720p30.mp4'

check = False
while not check:
    try:
        frames = prep(filename, multiplier=15, t0=522000, count=581)
    except BlockingIOError:
        raise
