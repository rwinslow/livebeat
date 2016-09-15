import os
import re
import sys

from PIL import Image, ImageFilter
import imageio
import numpy as np
import pandas as pd
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

def downsample_image(img):
    return block_reduce(img, block_size=(4, 4, 1), func=np.mean)

def crop_rect(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2, :]

def blackout_middle(img, x1, x2):
    img[:, x1:x2, :] = 0
    return img

def segment_intervals(filename, basedir, start_sec=0, end_sec=0, seconds_between_frame_grabs=10):
    """Generate timecodes for game and non-game segments"""
    # Get video id
    video_id = re.findall('v\d+', filename)[0]

    # Open file handle
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    nframes = meta['nframes']
    frames_to_get = np.arange(start_sec, end_sec, seconds_between_frame_grabs) * fps

    # Check frames
    for i in frames_to_get:
        try:
            img = vid.get_data(i)
        except:
            raise

        # Downlsample full image
        downsampled = downsample_image(img)

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

        # Write full frame and header frame
        imageio.imwrite(os.path.join(basedir, 'full_{}_s{}.png'.format(video_id, int(i/fps))), img)
        imageio.imwrite(os.path.join(basedir, 'pl_status_{}_s{}.png'.format(video_id, int(i/fps))), pl_status)
        imageio.imwrite(os.path.join(basedir, 'downsampled_{}_s{}.png'.format(video_id, int(i/fps))), downsampled)

    vid.close()

    return True

filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v83196893_720p30.mp4'
# filename = '/Volumes/Passport/LiveBeat/video/dota2ti_v83012529_720p30.mp4'

positive_timecodes = [
    # [5632, 5708],
    # [6872, 6959],
    # [7398, 7473],
    # [9001, 9077],
    # [9104, 9177],
    # [9249, 9319],
    # [9405, 9475],
    # [9643, 9717],
    # [9763, 9842],
    # [10017, 10087],
    # [10242, 10313],
    # [10590, 10663],
    # [11006, 11076],
    # [11876, 11949],
    # [12054, 12215],
    # [15281, 15354],
    # [15400, 15474],
    # [15890, 15960],
    # [16751, 16821],
    # [17045, 17115],
    # [17172, 17251],
    # [19289, 19365],
    # [19388, 19464],
    # [19692, 19771],
    # [19888, 19980],
    # [24511, 24586],
    # [25733, 25803],
    # [27941, 28011],
    # [28184, 28262],
    # [28955, 29036],
    # [29071, 29158],
    # [29208, 29329],
    # [29340, 29412],
    # [29422, 29538],
    # [29617, 29687],
    # [29766, 29836],
    # [30068, 30196]
]

negative_timecodes = [
    # [4589, 4691],
    [4693, 4876],
    [4878, 5214],
    [5216, 5447],
    [5449, 5691],
    [5693, 5849],
    [5879, 6888],
    [6890, 6931],
    [6933, 6948],
    [6950, 7457],
    [7459, 7460],
    [7463, 7469],
    [8935, 9060],
    [9067, 9163],
    [9165, 9227],
    [9229, 9308],
    [9310, 9463],
    [9465, 9701],
    [9709, 9822],
    [9832, 9964],
    [9966, 10076],
    [10078, 10300],
    [10303, 10601],
    [10604, 10649],
    [10654, 11064],
    [11066, 11780],
    [11782, 11859],
    [11861, 11935],
    [11937, 11938],
    [11940, 12123],
    [12125, 12187],
    [12190, 12193],
    [12195, 12196],
    [12199, 12204],
    [14935, 14992],
    [14994, 15340],
    [15342, 15343],
    [15345, 15461],
    [15465, 15949],
    [15951, 15985],
    [16311, 16400],
    [16402, 16810],
    [16812, 17103],
    [17105, 17231],
    [18891, 19112],
    [19114, 19289],
    [19291, 19310],
    [19313, 19347],
    [19349, 19353],
    [19355, 19447],
    [19452, 19453],
    [19455, 19697],
    [19699, 19751],
    [19757, 19760],
    [19762, 19946],
    [19960, 19964],
    [19966, 19969],
    [24467, 24570],
    [24573, 24575],
    [24577, 24696],
    [24698, 24717],
    [24719, 25177],
    [25179, 25270],
    [25278, 26087],
    [27977, 28000],
    [28002, 28242],
    [28253, 29014],
    [29017, 29025],
    [29027, 29130],
    [29132, 29138],
    [29140, 29141],
    [29146, 29147],
    [29149, 29267],
    [29269, 29270],
    [29277, 29292],
    [29294, 29295],
    [29300, 29301],
    [29305, 29306],
    [29308, 29312],
    [29314, 29315],
    [29317, 29318],
    [29320, 29399],
    [29402, 29480],
    [29491, 29492],
    [29494, 29526],
    [29528, 29675],
    [29677, 29825],
    [29827, 30131],
    [30134, 30175],
    [30184, 30185],
    [30187, 30316]
 ]

# basedir = '/Volumes/Passport/LiveBeat/video/interesting_v83196893'
# for interval in positive_timecodes:
#     print('Segmenting', interval)
#     segment_intervals(filename, basedir, interval[0], interval[1], 2)

basedir = '/Volumes/Passport/LiveBeat/video/uninteresting_v83196893'
for interval in negative_timecodes:
    print('Segmenting', interval)
    segment_intervals(filename, basedir, interval[0], interval[1], 10)
