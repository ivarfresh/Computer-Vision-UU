import os
import cv2 as cv

from shared.Calibrator import Calibrator

# Obtain intrinsic camera parameters
for cam in range(1, 5):
    intrinsics_path = os.path.join('data', f'cam{cam}', 'intrinsics.avi')
    vid = cv.VideoCapture(intrinsics_path)

    frames = []
    count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            # cv.imshow('Intrinsics', frame)
            frames.append(frame)
            # Advance 250 frames == 5 seconds (video is 50fps)
            count += 250
            vid.set(cv.CAP_PROP_POS_FRAMES, count)

        else:
            vid.release()
            break
    
    # Board and cell size adapted from checkerboard.xml
    calibrator = Calibrator(frames=frames, board_size=(8, 6), cell_size=115)
    calibrator.calibrate(recalibrate=True, save=True, savename=os.path.join('data', f'cam{cam}', 'intrinsics.npz'))
    cv.waitKey(0)