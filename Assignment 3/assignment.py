import glm
import numpy as np
import cv2 as cv
from sklearn import mixture
import matplotlib.pyplot as plt
from shared.VoxelCam import VoxelCam


# Global variables
block_size = 1.0
color_models = []


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([255.0, 255.0, 255.0])
    return data, colors


def set_voxel_positions(vc, path, path_colors):
    # Gets voxel locations from voxel reconstructor
    voxels, colors, parse_frame = vc.next_frame()

    # Translate voxel data (ignore height when clustering voxels)
    temp = [[voxel[0], 0, voxel[2]] for voxel in voxels]

    # Find clusters

    '''
    Use parameter 'attempt'
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 4
    _, labels, centers = cv.kmeans(np.float32(temp), k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # Divide clusters over unique lists and ignore voxels projected on the person's head or legs
    cluster_1 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 0 if (12 < voxels[i][1] < 27)]
    cluster_2 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 1 if (12 < voxels[i][1] < 27)]
    cluster_3 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 2 if (12 < voxels[i][1] < 27)]
    cluster_4 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 3 if (12 < voxels[i][1] < 27)]
    clusters = [cluster_1, cluster_2, cluster_3, cluster_4]

    # Give a color to each voxel (4 options based on 4 clusters)
    # options = [(255,0,0), (0,255,0), (0,0,255), (255,0,255)]
    # colors = [options[labels[i][0]-1] for i in range(len(voxels))]


    '''
    Create a color model for each person (cluster) if no models have been created yet.
    If models have been initialized, assign the correct color to each person (cluster) based on
    color model similarities.
    '''
    if not color_models:
        create_models(clusters)
    else:
        # Get voxel labels based on K means clustering and color model matching
        color_labels = set_voxel_colors(clusters, parse_frame)

        options = [(255,0,0), (0,255,0), (0,0,255), (255,0,255)]

        # Divide voxels over clusters (including head and legs)
        temp_1 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 0]
        colors_1 = [options[color_labels[0]] for _ in range(len(temp_1))]

        temp_2 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 1]
        colors_2 = [options[color_labels[1]] for _ in range(len(temp_2))]

        temp_3 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 2]
        colors_3 = [options[color_labels[2]] for _ in range(len(temp_3))]

        temp_4 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 3]
        colors_4 = [options[color_labels[3]] for _ in range(len(temp_4))]

        # reorder colors based on model, then add to path using centers
        colors_p = np.array(options)[np.array(color_labels)].tolist()
        path.extend(centers.tolist())
        path_colors.extend(colors_p)

        # combine all
        voxels = temp_1 + temp_2 + temp_3 + temp_4 
        colors = colors_1 + colors_2 + colors_3 + colors_4 

    return voxels, colors, path, path_colors


def get_cam_positions(vc):
    # Determines camera positions from voxel reconstructors rvec/tvec values
    positions = []
    for cam in vc.cams:
        # Calculate rotation matrix and determine position
        rmtx = cv.Rodrigues(cam.rvec)[0]
        pos = -rmtx.T * np.matrix(cam.tvec)

        # Swap Y and Z
        pos[[1, 2]] = pos[[2, 1]]

        # Take absolute of height variable (calibration had -z as positive height)
        pos[1] = abs(pos[1])
        # Divide by 10 to get camera at correct position (given resolution=50)
        positions.append(pos/75)
    return positions, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def get_cam_rotation_matrices(vc):
    # Calculate correct camera rotations based on rotation matrix
    rotations = []
    for cam in vc.cams:
        I = np.identity(4)
        rmtx = cv.Rodrigues(cam.rvec)[0]
        I[0:3,0:3] = rmtx

        glm_mat = glm.mat4(I)

        # Rotate 2nd and 3rd dimensions by 90 degrees
        glm_mat = glm.rotate(glm_mat, glm.radians(90), (0, 1, 1))
        rotations.append(glm_mat)
    return rotations


################################################################################

'''
Function that creates a color model out of one frame given 3D voxel coordinates.
'''
def create_models(clusters):
    global color_models

    cams = ['cam1','cam2','cam3','cam4']
    for cam in cams:
        cap = cv.VideoCapture('data/video/'+cam+'.avi')
        _,frame = cap.read()
        color_models.append(voxels_hsv(clusters, frame, cam))


'''
Obtain the 2D image plane coordinates of each 3D voxel of each cluster.
Additionally, find the hsv values of these 2D coordinates and store them for
each cluster.
'''
def voxels_hsv(clusters, frame, cam):
    # Load the (inversed) table only for Cam2
    table_path = 'data/table_inv/'+cam+'.npz'
    with np.load(table_path, allow_pickle=True) as f_table:
        table = f_table['table'].tolist()

    # Find the 2D image plane coordinates corresponding to the 3D voxels
    imgpts = []
    for cluster in clusters:
        pts = []
        for pt in cluster:
            pt = tuple([int(pt[0]),int(pt[1]),int(pt[2])])
            if pt in table:
                pts.append(table[pt])
        imgpts.append(pts)

    # Visualise the retrieved 2D image points by drawing them on a copy of the original frame
    # copy = np.copy(frame)
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    # for i in range(4):
    #     for pt in imgpts[i]:
    #         cv.circle(copy, (int(pt[0]), int(pt[1])), 2, colors[i], thickness=cv.FILLED)
    # cv.imwrite('images/clusters.jpg', copy)

    # Convert frame from bgr to rbg
    # hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Create a list of rbg values of the 2D coordinates corresponding to the 3D voxels for each cluster
    cluster_frames = []
    # count = 1
    for pts in imgpts:
        # print(len(pts))
        i, j = 0, 0
        size = int(np.sqrt(len(pts))) + 1
        cluster = np.zeros([size, size, 3], dtype=np.uint8)
        cluster.fill(255)
        for pt in pts:
            cluster[i][j] = frame[int(pt[1])][int(pt[0])]
            i += 1
            if i == size:
                i = 0
                j += 1
        # cv.imwrite(f'images/color_grid{count}.jpg', cluster)
        # count += 1
        cluster_frames.append(cluster)

    histograms = []

    # Create a histogram for each cluster (person)
    cluster_frames = [cv.cvtColor(cluster, cv.COLOR_BGR2HSV) for cluster in cluster_frames]
    index = 1
    for cluster in cluster_frames:
        hist_list = histogram(cluster, index)
        histograms.append(hist_list)
        index += 1

    # Get hsv values of 2D voxel coordinates
    # hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # cv.imwrite('images/hsv.jpg', hsv_frame)

    # Create a list of hsv values of the 2D coordinates corresponding to the 3D voxels
    # hsv_list = []
    # for pts in imgpts:
    #     hsv = []
    #     for pt in pts:
    #         hsv.append(hsv_frame[int(pt[1])][int(pt[0])])
    #     hsv_list.append(hsv)

    return histograms


'''
Calculate histogram list (concatenation of each HSV channel)
'''
def histogram(cluster, index):
    hist_list = []
    # Do not take into account all-white pixels
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([cluster], [i], None, [254], [0, 254])
        hist_list.append(hist)
        # plt.plot(hist, color = col)
        # plt.xlim([0, 254])
    # plt.savefig(f'images/histogram_{index}.png')
    return hist_list


'''
Compare the hsv histograms of a new frame to each color model (of each person)
and decide which cluster should get which color (based on distance/similarity).
'''
def set_voxel_colors(clusters, parse_frame):

    histograms = []
    cams = ['cam1','cam2','cam3','cam4']
    for idx in range(4):
        histograms.append(voxels_hsv(clusters, parse_frame[idx], cams[idx]))

    all_distances = []
    for h in histograms: # 4 cams
        for hist in h: # 4 modellen
            temp_dist = []

            for m in color_models: # 4 cams
                for model in m: # 4 modellen
                    curr_dist = 0
                    for i in range(3):
                        curr_dist += cv.compareHist(hist[i], model[i], cv.HISTCMP_CORREL)
                    temp_dist.append(curr_dist/3)
            all_distances.append(temp_dist)

    # modulo 4 so every index of every cam is still 0,1,2,3
    label_1 = all_distances[0].index(max(all_distances[0])) % 4
    label_2 = all_distances[1].index(max(all_distances[1])) % 4
    label_3 = all_distances[2].index(max(all_distances[2])) % 4
    label_4 = all_distances[3].index(max(all_distances[3])) % 4
    labels = [label_1, label_2, label_3, label_4]

    #Hungarian matching
    #from scipy.optimize import linear_sum_assignment
    #row_ind, col_ind = linear_sum_assignment(all_distances)
    #h_label_1 = col_ind[0]
    #print(f'h_label_1:{h_label_1}')
    #h_label_2 = col_ind[1]
    #h_label_3 = col_ind[2]
    #h_label_4 = col_ind[3]
    #labels = [h_label_1, h_label_2, h_label_3, h_label_4]

    return labels


'''
Function that draws a trace of voxel movement on a white 2D image
'''
def draw_trace(path, colors):
    options = [[255,0,0], [0,255,0], [0,0,255], [255,0,255]]

    red_id = [idx for idx, value in enumerate(colors) if value == options[0]]
    blue_id = [idx for idx, value in enumerate(colors) if value == options[1]]
    green_id = [idx for idx, value in enumerate(colors) if value == options[2]]
    pink_id = [idx for idx, value in enumerate(colors) if value == options[3]]

    red = [(path[idx][0], path[idx][2]) for idx in red_id]
    blue = [(path[idx][0], path[idx][2]) for idx in blue_id]
    green = [(path[idx][0], path[idx][2]) for idx in green_id]
    pink = [(path[idx][0], path[idx][2]) for idx in pink_id]
    scatters = [red, blue, green, pink]

    for i in range(4):
        options = [[1,0,0], [0,1,0], [0,0,1], [1,0,1]]
        path_len = len(scatters[i])
        x = [scatters[i][j][1] for j in range(path_len)]
        y = [scatters[i][j][0] for j in range(path_len)]
        c = [options[i] for _ in range(path_len)]
        plt.scatter(x, y, c=c)

    plt.savefig('images/trace.png')
