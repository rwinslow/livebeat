import cv2

vidcap = cv2.VideoCapture('./video/dota2ti_v82878048_360p30.avi')
success, image = vidcap.read()
count = 0
while success:
    success, image = vidcap.read()
    print(count)
