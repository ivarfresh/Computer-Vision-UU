import numpy as np
import cv2 as cv
import glob
import os

# Calibrator processes the images meant for calibration and calibrates the camera
class Calibrator:

    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    EPSILON = 0.01

    objpoints = []
    imgpoints = []

    def __init__(self, path=None, frames=None, board_size=(9,6), cell_size=24, intrinsics_path=None):
        # Constructor of Calibrator class, finds paths of images that should be used
        self.path = path
        if path is not None:
            self.images = glob.glob(path)

        if frames is not None:
            self.frames = frames

        self.intrinsics_path = intrinsics_path
        if self.intrinsics_path is not None:
            with np.load(os.path.join(intrinsics_path)) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']

        self.click_coords = []
        self.manual_corners = []
        self.prev_ret = 1
        self.cell_size = cell_size

        self.board_size = board_size
        self.objp = np.zeros((self.board_size[1]*self.board_size[0], 3), np.float32)
        # Adjust object point size for cell size, spaces each square by cell_size
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.cell_size

    def interpolate(self, start_coord, end_coord, stepsize):
        # Linearly interpolate between two coordinates, finding stepsize
        # amount of equidistant points between start_coord and end_coord
        x = np.linspace(start_coord[0], end_coord[0], stepsize)
        y = np.linspace(start_coord[1], end_coord[1], stepsize)

        return list(zip(x, y))

    def transform_warped(self, points):
        # Warps part of an image to create a top-down perspective

        points = np.array(points, dtype=np.float32)
        # Order is top-left, top-right, bottom-right, bottom-left
        top_left = points[0]
        top_right = points[1]
        bottom_right = points[2]
        bottom_left = points[3]

        # Calculate width as maximum of horizontal lines in selected shape (Euclidean)
        widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Calculate height as maximum of vertical lines in selected shape (Euclidean)
        heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Points that the source should be transformed to
        dst_points = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype=np.float32)

        # Transformation matrix
        M = cv.getPerspectiveTransform(points, dst_points)
        warped = cv.warpPerspective(self.frame, M, (maxWidth, maxHeight))
        return M, warped

    def click_event(self, event, x, y, flags, param):
        # Event triggered on click during manual annotation of checkersboard corners
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'Clicked at coordinate ({x},{y})')
            self.click_coords.append((x, y))
            cv.circle(self.frame, (x, y), 2, (0, 0, 255), thickness=cv.FILLED)
            cv.imshow('img', self.frame)

            if len(self.click_coords) == 4:
                # Use perspective transform
                M, warped = self.transform_warped(self.click_coords)
                warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

                # Linearly interpolate from corners of warped part of image, these corners should be the corners of the chessboard
                left_vert = self.interpolate((0, 0), (0, warped.shape[0]), self.board_size[1])
                right_vert = self.interpolate((warped.shape[1], 0), (warped.shape[1], warped.shape[0]), self.board_size[1])
                corners = []
                for left, right in zip(left_vert, right_vert):
                    corners.extend(self.interpolate(left, right, self.board_size[0]))

                corners = np.expand_dims(np.array(corners, dtype=np.float32), 1)
                
                # Before inversely transforming the corners back to original 2d space, find better locations using subpixels
                if self.intrinsics_path is None:
                    corners = cv.cornerSubPix(warped_gray, corners, (11, 11), (-1, -1), self.CRITERIA)

                # Inverse perspective transformation
                _, IM = cv.invert(M)
                points_transformed = cv.perspectiveTransform(corners, IM)

                for point in points_transformed:
                    cv.circle(self.frame, (int(point[0][0]), int(point[0][1])), 2, (0, 0, 255), thickness=cv.FILLED)

                cv.imshow('img', self.frame)

                self.click_coords = []

                self.imgpoints.append(points_transformed)
                self.objpoints.append(self.objp)

    def process_images(self):
        # Processes all images at given location, automatically or manually detecting chessboard corners, supports filenames for images and a list of frames for video
        if self.path is not None:
            for idx, fname in enumerate(self.images):
                frame = cv.imread(fname)
                self.process_image(frame, idx)
        else:
            for idx, frame in enumerate(self.frames):
                self.process_image(frame, idx)
        cv.destroyAllWindows()

    def process_image(self, frame, index):
        # Process a single frame, finding chessboard corners and calibrating the camera
        self.frame = frame
        self.gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(self.gray, self.board_size, None)

        ret = False ##### FORCE HANDMATIG
        if ret:
            # Corners are found
            self.objpoints.append(self.objp)
            corners_subpix = cv.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.CRITERIA)
            self.imgpoints.append(corners_subpix)

            cv.drawChessboardCorners(frame, self.board_size, corners_subpix, ret)
            cv.imshow('img', frame)
            cv.waitKey(5)
        else:
            # Corners not found, manually request corners
            print(f'Did not find corners for frame: {index}')

            cv.imshow('img', frame)
            cv.setMouseCallback('img', self.click_event)
            cv.waitKey(0)
        
        # Dont use reprojection error to recalibrate if intrinsics is given
        if self.intrinsics_path is not None:
            return

        # Calibrate camera and only use new image if projection error does not decrease by more than epsilon
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        print(f"Re-projection error: {self.ret:.4}")
        if self.ret - self.prev_ret > self.EPSILON:
            print(f'Frame {index} made projection error worse by: {self.ret - self.prev_ret:.4}')
            self.objpoints.pop()
            self.imgpoints.pop()
        self.prev_ret = self.ret
    
    def calibrate(self, recalibrate=False, save=True, savename='resources/calibration.npz'):
        # Loads existing calibration or calibrates the camera
        if not recalibrate:
            with np.load(savename) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']
        else:
            self.process_images()
            if self.intrinsics_path is None:
                # Final calibration with only improving frames
                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
                if save:
                    np.savez(savename, ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
            else:
                # Use objp and first entry in imgpoints (all corners found in the frame) to calibrate extrinsics
                _, rvec, tvec = cv.solvePnP(self.objp, self.imgpoints[-1], self.mtx, self.dist)
                # Use solvePnP to find extrinsics for img/cam
                # Draw frame axes (x, y, z) directions on frame
                cv.drawFrameAxes(self.frame, self.mtx, self.dist, rvec, tvec, self.cell_size * (self.board_size[1] - 1), thickness=2)
                cv.imshow('img', self.frame)
                cv.waitKey(0)
                if save:
                    # Save intrinsics that were used plus newly calculated extrinsics
                    np.savez(savename, ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs, rvec=rvec, tvec=tvec)