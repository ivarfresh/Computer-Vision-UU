import glm
import numpy as np
import cv2 as cv
from sklearn import mixture
import matplotlib.pyplot as plt
from shared.VoxelCam import VoxelCam
from scipy.optimize import linear_sum_assignment


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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 4
    _, labels, centers = cv.kmeans(np.float32(temp), k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # Divide clusters over unique lists and ignore voxels projected on the person's head or legs
    cluster_1 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 0 if (12 < voxels[i][1] < 26)]
    cluster_2 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 1 if (12 < voxels[i][1] < 26)]
    cluster_3 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 2 if (12 < voxels[i][1] < 26)]
    cluster_4 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 3 if (12 < voxels[i][1] < 26)]
    clusters = [cluster_1, cluster_2, cluster_3, cluster_4]

    '''
    Create a color model for each person (cluster) if no models have been created yet.
    If models have been initialized, assign the correct color to each person (cluster) based on
    color model similarities.
    '''
    if not color_models:
        create_models(vc, clusters)
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
        voxels = temp_1 + temp_2 + temp_3 + temp_4 + path
        colors = colors_1 + colors_2 + colors_3 + colors_4 + path_colors

    return voxels, colors, path, path_colors

    '''
    Create path on the floor grid
    FIX when colors for each individual is fixed
    '''

    # # create path
    if path == []:
        path = centers.tolist()
        path_colors = [[255,0,0],[255,0,0],[255,0,0],[255,0,0]]
    else:
        path.extend(centers.tolist())
        path_colors.extend([[255,0,0],[255,0,0],[255,0,0],[255,0,0]])

    # add path to voxel space
    for voxel in path:
        voxels.append(voxel)
    colors.extend(path_colors)

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

def EM_models(person_color):
    '''
    Create and train color models. Also get mean of the prediction which can be used for comparison. 
    '''
    # the number of cluster's
    N = 3

    #create + train EM model
    em = cv.ml.EM_create()
    em.setClustersNumber(N)
    em.trainEM(person_color)
    return em


def predict(em, samples):
    '''
    Function to predict the online models based on offline models and new frames
    '''
    overall_loghood = 0
    for sample in samples:
        (loglikelihood, _), _ = em.predict2(sample)
        overall_loghood += loglikelihood
    return overall_loghood


'''
Function that creates a color model out of one frame given 3D voxel coordinates.
'''
def create_models(vc, clusters):
    global color_models

    frame = cv.imread('images/frame.jpg')
    copy = np.copy(frame)

    _, _, parse_frame = vc.next_frame()

    
    person_1_off, person_2_off, person_3_off, person_4_off = voxels_hsv(clusters, frame)
    print(f'person off len: {len(person_1_off) + len(person_2_off) + len(person_3_off) + len(person_4_off)}')

    #create + train EM offline model for all persons
    em1 = EM_models(person_1_off)
    em2 = EM_models(person_2_off)
    em3 = EM_models(person_3_off)
    em4 = EM_models(person_4_off)

    color_models = [em1, em2, em3, em4]


'''
Compare each cluster to the created color models to determine which cluster
belongs to which person.
'''
def set_voxel_colors(clusters, parse_frame):
    '''
    Compare the hsv histogram of a new frame to each color model (of each person)
    and decide which cluster should get which color (based on distance/similarity).
    Hopefully, we end up with a list of colors that can be passed back.
    '''

    #construct online voxel persons
    person_1_on, person_2_on, person_3_on, person_4_on = voxels_hsv(clusters, parse_frame)
    person_on_list = [person_1_on, person_2_on, person_3_on, person_4_on]

    em1, em2, em3, em4 = color_models

    #predict
    overall_log1_1 = predict(em1, person_1_on)
    overall_log1_2 = predict(em1, person_2_on)
    overall_log1_3 = predict(em1, person_3_on)
    overall_log1_4 = predict(em1, person_4_on)
    
    overall_log2_1 = predict(em2, person_1_on)
    overall_log2_2 = predict(em2, person_2_on)
    overall_log2_3 = predict(em2, person_3_on)
    overall_log2_4 = predict(em2, person_4_on)
  
    overall_log3_1 = predict(em3, person_1_on)
    overall_log3_2 = predict(em3, person_2_on)
    overall_log3_3 = predict(em3, person_3_on)
    overall_log3_4 = predict(em3, person_4_on)
    
    overall_log4_1 = predict(em4, person_1_on)
    overall_log4_2 = predict(em4, person_2_on)
    overall_log4_3 = predict(em4, person_3_on)
    overall_log4_4 = predict(em4, person_4_on)

    #hungarian matching
    logs = np.array([[overall_log1_1, overall_log1_2, overall_log1_3, overall_log1_4],
                    [overall_log2_1, overall_log2_2, overall_log2_3, overall_log2_4],
                    [overall_log3_1, overall_log3_2, overall_log3_3, overall_log3_4],
                    [overall_log4_1, overall_log4_2, overall_log4_3, overall_log4_4]
                    ])


    #Hungarian matching
    row_ind, col_ind = linear_sum_assignment(logs)
    h_label_1 = col_ind[0]
    h_label_2 = col_ind[1]
    h_label_3 = col_ind[2]
    h_label_4 = col_ind[3]
    labels = [h_label_1, h_label_2, h_label_3, h_label_4]

    return labels


'''
Obtain the 2D image plane coordinates of each 3D voxel of each cluster.
Additionally, find the hsv values of these 2D coordinates and store them for
each cluster.
'''
def voxels_hsv(clusters, frame):
    # Load the (inversed) table only for Cam2
    table_path = 'data/table_inv/cam2.npz'
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
    copy = np.copy(frame)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    for i in range(4):
        for pt in imgpts[i]:
            cv.circle(copy, (int(pt[0]), int(pt[1])), 2, colors[i], thickness=cv.FILLED)
    cv.imwrite('images/clusters.jpg', copy)

    # Get hsv values of 2D voxel coordinates
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imwrite('images/hsv.jpg', hsv_frame)

    # Create a list of hsv values of the 2D coordinates corresponding to the 3D voxels
    hsv_list = []
    for pts in imgpts:
        hsv = []
        for pt in pts:
            hsv.append(hsv_frame[int(pt[1])][int(pt[0])])
        hsv_list.append(hsv)

    RGB_list = []
    for pts in imgpts:
        RGB = []
        for pt in pts:
            RGB.append(frame[int(pt[1])][int(pt[0])])
        RGB_list.append(RGB)
    print(f'len(RGB_list):{len(RGB_list[0]) + len(RGB_list[0]) + len(RGB_list[0]) + len(RGB_list[0])}')
    
    #save color values of voxels corresponding to persons
    person_1_RGB = np.float32(RGB_list[0])
    person_2_RGB = np.float32(RGB_list[1])
    person_3_RGB = np.float32(RGB_list[2])
    person_4_RGB = np.float32(RGB_list[3])

    return person_1_RGB, person_2_RGB, person_3_RGB, person_4_RGB



'''
Function that draws a trace of voxel movement on a white 2D image
'''
def draw_trace(path, colors):
    # Create white image
    img = np.zeros([128,128,3],dtype=np.uint8)
    img.fill(255)

    # Add path coordinates
    for i in range(len(path)):
        # Transform 3D to 2D coordinates and translate to deal with negative numbers
        x = int(path[i][0]) + 50
        y = int(path[i][2]) + 50
        img[x][y] = colors[i]
        # Increase resolution
        # img[x+1][y] = path_colors[i]
        # img[x][y+1] = path_colors[i]
        # img[x+1][y+1] = path_colors[i]
        # img[x-1][y] = path_colors[i]
        # img[x][y-1] = path_colors[i]
        # img[x-1][y-1] = path_colors[i]

    cv.imwrite('images/trace.jpg', img)
