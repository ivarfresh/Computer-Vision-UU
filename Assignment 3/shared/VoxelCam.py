from shared.BackgroundSubtractor import BackgroundSubtractor
import os, numpy as np, cv2 as cv

BASE_PATH = os.path.join('data')

# VoxelCam class handles the cams and their videos, configuration, and calibration files
class VoxelCam():

    def __init__(self, idx):
        self.idx = idx
        cam_path = os.path.join(BASE_PATH, f'cam{idx}')
        background_path = os.path.join(f'data/background/cam{idx}.avi')
        video_path = os.path.join(f'data/video/cam{idx}.avi')
        config_path = os.path.join(f'data/config/cam{idx}.npz')

        self.subtractor = BackgroundSubtractor(background_path)
        self.fg = None

        self.video = cv.VideoCapture(video_path)
        with np.load(config_path) as calibration:
            self.mtx = calibration['mtx']
            self.dist = calibration['dist']
            self.rvec = calibration['rvec']
            self.tvec = calibration['tvec']

    # Advance the video of this cam by one frame and return it
    def next_frame(self):
        if self.video.isOpened():
            ret, self.frame = self.video.read()
            if ret:
                # Get foreground for new frame and create a bitwise XOR representation between it and the previous foreground
                fg_tmp = self.subtractor.get_foreground(self.frame)
                self.xor = fg_tmp if self.fg is None else cv.bitwise_xor(fg_tmp, self.fg)
                self.fg = fg_tmp
                return ret, self.frame
            else:
                # Reset video to first frame, looping
                _ = self.video.set(cv.CAP_PROP_POS_FRAMES, 0)
                self.next_frame()
        else:
            return False, None

    
    # Since objects like cv.VideoCapture and BackgroundSubtractorKNN are not pickle-able,
    # create subset of info needed for creating the lookup table and return it
    def get_info(self):
        return {
            'mtx': self.mtx,
            'dist': self.dist,
            'rvec': self.rvec,
            'tvec': self.tvec,
            'idx': self.idx
        }
