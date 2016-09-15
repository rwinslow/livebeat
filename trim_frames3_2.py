import os
import re
import time

import imageio
import numpy as np
from PIL import Image, ImageFilter


def crop_rect(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2, :]

def blackout_middle(img, x1, x2):
    img[:, x1:x2, :] = 0
    return img

def trim_to_frames(filename, codec='ffmpeg', multiplier=1, t0=0, count=0):
    """Cut video to 1 frame per second and save images"""

    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, codec)

    # Get metadata
    img = vid.get_data(1)
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    frames = np.arange(t0, meta['nframes'], fps * multiplier)
    interval = fps * multiplier

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

    vid.close()

    # Shop button
    h, w, c = img.shape
    x1 = int(w * .94)
    x2 = int(w * 1)
    y1 = int(h * .814)
    y2 = int(h * .85)
    tlx = x1
    tly = y1
    span_x = x2-x1
    span_y = y2-y1

    string = 'ffmpeg -i {} -vf "fps=1,crop={}:{}:{}:{}" ./test2/full_%07d.png'.format(filename, span_x, span_y, tlx, tly)
    print(string)
    os.system(string)

    return True

filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v82878048_720p30.mp4'
result = trim_to_frames(filename, multiplier=15)
