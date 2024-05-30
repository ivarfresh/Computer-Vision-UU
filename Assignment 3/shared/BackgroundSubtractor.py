import cv2 as cv
import numpy as np

# The BackgroundSubtractor class handles subtracting background (or creating foregrounds) from images
class BackgroundSubtractor:

    def __init__(self, path):
        # Use OpenCV's KNN background subtractor
        self.subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True)
        vid = cv.VideoCapture(path)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                # Learn from every frame
                self.subtractor.apply(frame, learningRate=0.5)
            else:
                vid.release()
                break
        pass

    def get_foreground(self, frame):
        # Do not learn as much from actual video frames
        fg = self.subtractor.apply(frame, learningRate=0.00001)

        # Remove detected shadows
        fg[fg == 127] = 0

        # Applying the morphologic closing operation (dilating, then eroding) gave results
        # that helped denoising the edges of the foreground without losing much detail, if any
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel, iterations=1)

        # Find the biggest contour in the image

        contours, hierarchy = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # biggest_contour = max(contours, key=cv.contourArea)

        # # Draw filled version of biggest contour onto empty mask, resulting in final foreground
        # mask = np.zeros(fg.shape, np.uint8)
        # cv.drawContours(mask, [biggest_contour], 0, 255, -1)
        cv.drawContours(fg, contours, -1, (0,255,0), 3)
        

        return fg