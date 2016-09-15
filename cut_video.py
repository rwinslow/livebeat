import os
import re
import time

import imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

def cut_video(video_file, timecodes_file, output_file):
    """Cut video to 1 frame per second and save images"""

    # Get video id
    video_id = re.findall('v\d+', video_file)[0]

    # Open file handle
    vid = imageio.get_reader(video_file, 'ffmpeg')

    # Get metadata
    meta = vid.get_meta_data()
    fps = int(meta['fps'])
    vid.close()

    # Make directories for storage
    basedir = '/Volumes/Passport/LiveBeat/video/cuts'
    try:
        os.makedirs(os.path.join(basedir))
    except FileExistsError:
        pass

    # Import timecode data
    timecodes = pd.read_csv(timecodes_file, names=['start', 'stop'])
    timecodes = timecodes.astype(int)

    # Smear timecodes to remove anything < 30 seconds long
    smear_tc = timecodes[timecodes['stop'] - timecodes['start'] >= 30000].reset_index(drop=True)

    # ffmpeg -ss 00:00:30 -i orginalfile -t 00:00:05 -vcodec copy -acodec copy newfile


    # Run through smeared timecodes and flip switches on correct seconds of video
    cmd = ""
    for i in smear_tc.index:
        row = smear_tc.iloc[i][['start', 'stop']]
        start = int(row[0]/fps)
        stop = int(row[1]/fps)
        # cmd += "[0:v]trim={}:{},setpts=PTS-STARTPTS[v{}];[0:a]atrim={}:{},asetpts=PTS-STARTPTS[a{}];".format(start, stop, i, start, stop, i)
        # cmd += '-ss {} -t {} -i "{}" '.format(start, stop-start, video_file)
        full_command = 'ffmpeg -ss {} -i {} -t {} -strict -2 {}'.format(start, video_file, stop-start, output_file+str(i)+'.mp4')
        print(full_command)
        os.system(full_command)
    #
    # filter_complex = ""
    # for v in range(i):
    #     filter_complex += '[{}]'.format(v)
    # filter_complex += 'concat=n={}'.format(i+1)
    # for v in range(i):
    #     filter_complex += ':v={}:a={}'.format(v, v)
    # filter_complex = '-filter_complex "{}"'.format(filter_complex)

    # full_filter = cmd + filter_close
    # full_command = '''ffmpeg -i {} -filter_complex "{}" -map "out" {}'''.format(video_file, full_filter, output_file)
    # full_command = 'ffmpeg {} {} -strict -2 {}'.format(cmd, filter_complex, output_file)

    # full_command = 'ffmpeg -ss 1:00 -t 5 -i {} -ss 2:00 -t 5 -i {} -filter_complex "[0][1]concat=n=2:v=1:a=1" -strict -2 {}'.format(video_file, video_file, output_file)

    print(full_command)
    os.system(full_command)

    return True

video_file = '/Volumes/Passport/LiveBeat/video/dota2ti_v83196893_720p30.mp4'
timecodes_file = './timecodes/timecodes_v83196893.csv'
output_file = '/Volumes/Passport/LiveBeat/video/dota2ti_v83196893_720p30_clipped.mp4'
cut_video(video_file, timecodes_file, output_file)


# ffmpeg -i utv.ts -filter_complex \
# "[0:v]trim=duration=30[av];[0:a]atrim=duration=30[aa];\
#  [0:v]trim=start=40:end=50,setpts=PTS-STARTPTS[bv];\
#  [0:a]atrim=start=40:end=50,asetpts=PTS-STARTPTS[ba];\
#  [av][bv]concat[cv];[aa][ba]concat=v=0:a=1[ca];\
#  [0:v]trim=start=80,setpts=PTS-STARTPTS[dv];\
#  [0:a]atrim=start=80,asetpts=PTS-STARTPTS[da];\
#  [cv][dv]concat[outv];[ca][da]concat=v=0:a=1[outa]" -map [outv] -map [outa] out.ts
