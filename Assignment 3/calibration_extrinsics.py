import os
import cv2 as cv

from shared.Calibrator import Calibrator

# Calibrate extrinsics for every camera
for cam in range(1, 5):
    base_path = os.path.join('data', f'cam{cam}')
    extrinsics_path = os.path.join(f'data/extrinsics/cam{cam}.avi')
    intrinsics_path = os.path.join(base_path, 'intrinsics.npz')

    vid = cv.VideoCapture(extrinsics_path)

    frames = []
    ret, frame = vid.read()
    if ret:
        frames.append(frame)
    else:
        vid.release()

    # Board and cell size adapted from checkerboard.xml
    calibrator = Calibrator(frames=frames, board_size=(8, 6), cell_size=115, intrinsics_path=intrinsics_path)

    # When calibrating extrinsics, calibrator uses perspectiveTransform
    calibrator.calibrate(recalibrate=True, save=True, savename=os.path.join(f'data/config/cam{cam}.npz'))

    # Added due to issues with lists inside calibrator remaining in new loop
    del calibrator
    vid.release()
    cv.destroyAllWindows()