import os
import subprocess

audiobase = '/Volumes/Passport/LiveBeat/audio'
videobase = '/Volumes/Passport/LiveBeat/video'

video_id = '83400929'
videofile = os.path.join(videobase, 'dota2ti_v{}_720p30.mp4'.format(video_id))
audiofile = os.path.join(audiobase, 'a{}.wav'.format(video_id))

command = "ffmpeg -i {} -ac 2 -ar 8000 -vn {}".format(videofile, audiofile)

subprocess.call(command, shell=True)
