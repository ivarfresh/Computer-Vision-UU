import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, BASE_PATH
import multiprocessing as mp
import os
from collections import defaultdict

# Resolution to use when generating the voxel model
RESOLUTION = 50

# The VoxelReconstruct class handles calculating  the lookup tables for individual VoxelCams
# and gathering the voxel, color combinations for the next frame.
class VoxelReconstructor():

    def __init__(self, create_table=False):
        # Create VoxelCam instances and pre-load their pickle-able information sets
        self.cams = []
        self.cam_infos = []

        for cam in range(1, 5):
            vcam = VoxelCam(cam)
            self.cams.append(vcam)
            self.cam_infos.append(vcam.get_info())

        self.cam_amount = len(self.cams)

        # (X, Y, Z) -> count, color dicts
        self.voxels = defaultdict(lambda: [False] * self.cam_amount)
        self.colors = defaultdict(lambda: [[0, 0, 0]] * self.cam_amount)

        if create_table:
            # Parallelized calculation of lookup table
            with mp.Pool(len(self.cams)) as p:
                results = p.map(calc_table, self.cam_infos)

            for result in results:
                self.cams[result[0] - 1].table = result[1]

            # Save tables to disk
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'table/cam{idx + 1}.npz')
                np.savez(table_path, table=cam.table)

                keys = list(cam.table.keys())
                values = []
                for voxel in cam.table.values():
                    values.append((int(voxel[0][0]-(int(RESOLUTION)/2)), int(-voxel[0][2]+RESOLUTION), int(voxel[0][1]-(int(RESOLUTION)/2))))

                values = list(values)
                table_inv = dict(zip(values,keys))

                table_path = os.path.join(BASE_PATH, f'table_inv/cam{idx + 1}.npz')
                np.savez(table_path, table=table_inv)



        else:
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'table/cam{idx + 1}.npz')
                with np.load(table_path, allow_pickle=True) as f_table:
                    cam.table = f_table['table']

        ########################################################################
        # all_voxels = []
        # for x in self.cams:
        #     all_voxels.append(list(x.table.values())) ## values is 3D, key is 2D
        #     print(list(x.table.values()))
        #
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85) #criteria
        # k = 4 # Choosing number of cluster
        # retval, labels, centers = cv.kmeans(np.float32(all_voxels), k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # print(retval)
        # print(labels)
        # print(centers)
        ########################################################################

    # Selects the changed voxels + colors for the next frame
    def next_frame(self):
        next_voxels = []
        next_colors = []

        parse_frame = []

        # For every cam/view, advance it one frame and receive the changed pixels
        for cam in self.cams:
            ret, frame = cam.next_frame()
            parse_frame.append(frame)
            if not ret:
                return
            changed = cam.xor.nonzero()

            # print(f'{cam.idx} Changed pixels: {len(changed[0])}')

            # For every changed foreground pixel, set it's show-value (True/False) for the cam it is shown on
            # also add the color of the foreground pixel for the current camera
            for pix_y, pix_x in zip(changed[0], changed[1]):
                coord = (pix_x, pix_y)
                for voxel in cam.table[coord]:
                    # Set show-value of voxel based on foreground value of changed pixel
                    self.voxels[voxel][cam.idx-1] = cam.fg[pix_y, pix_x] != 0

                    if cam.fg[pix_y, pix_x] != 0:
                        self.colors[voxel][cam.idx-1] = cam.frame[pix_y, pix_x]

        # For all the voxel, color combinations, add those that occurred in all cameras
        for voxel, check in self.voxels.items():
            if all(check):
                # Add voxel including offsetting due to calibration and to place it in the middle of the plane
                next_voxels.append([voxel[0]-(int(RESOLUTION)/2), -voxel[2]+RESOLUTION, voxel[1]-(int(RESOLUTION)/2)])

                # Divide by 100 to get color in [0, 1] interval
                # color = np.mean(np.array(self.colors[voxel]), axis=0) / 100

                # BGR to RGB (for OpenGL)
                # color[[0, 2]] = color[[2, 0]]

                next_colors.append([255,255,255])
                # next_colors.append(color)

        return next_voxels, next_colors, parse_frame

# Create (X, Y) -> [(X, Y, Z), ...] dictionary
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    print(f'{cam["idx"]} Calculating table')
    steps_c = complex(RESOLUTION)
    # Create evenly spaced grid centered around the subject, using n = RESOLUTION steps.
    grid = np.float32(np.mgrid[-1000:3000:steps_c, -3000:2000:steps_c, -3000:0:steps_c])
    grid_t = grid.T.reshape(-1, 3)

    print(f'{cam["idx"]} Projecting points')
    # Project 3d voxel locations to 2d
    proj_list = cv.projectPoints(grid_t, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

    ########################################################################################
    # visualize voxel space
    # vid = cv.VideoCapture(f'data/video/cam{cam["idx"]}.avi')
    # while vid.isOpened():
    #     ret, frame = vid.read()
    #     if ret:
    #         for x in proj_list:
    #             cv.circle(frame, (int(x[0][0]), int(x[0][1])), 2, (0, 0, 255), thickness=cv.FILLED)
    #         cv.imshow('img', frame)
    #         if cv.waitKey(1) == ord('q'):
    #             break
    ########################################################################################

    # Create table of (X, Y, Z, 2) shape containing 2d coordinates
    table = np.int_(proj_list).reshape((RESOLUTION, RESOLUTION, RESOLUTION, 2), order='F')

    # Create dictionary: (X,Y) -> [(X, Y, Z), (X, Y,Z), ...]
    table_d = defaultdict(list)
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            for z in range(RESOLUTION):
                table_d[tuple(table[x,y,z,:])].append((x,y,z))

    print(f'{cam["idx"]} Wrapping up')
    return (cam['idx'], table_d)
