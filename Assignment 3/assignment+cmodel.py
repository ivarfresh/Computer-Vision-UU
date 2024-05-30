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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 4
    _, labels, centers = cv.kmeans(np.float32(temp), k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # Divide clusters over unique lists and ignore voxels projected on the person's head or legs
    cluster_1 = [voxels[i] for i in range(len(voxels)) if labels[i] == 0 if (12 < voxels[i][1] < 26)]
    cluster_2 = [voxels[i] for i in range(len(voxels)) if labels[i] == 1 if (12 < voxels[i][1] < 26)]
    cluster_3 = [voxels[i] for i in range(len(voxels)) if labels[i] == 2 if (12 < voxels[i][1] < 26)]
    cluster_4 = [voxels[i] for i in range(len(voxels)) if labels[i] == 3 if (12 < voxels[i][1] < 26)]
    clusters = [cluster_1, cluster_2, cluster_3, cluster_4]

    '''
    Create a color model for each person (cluster) if no models have been created yet.
    If models have been initialized, assign the correct color to each person (cluster) based on
    color model similarities.
    '''
    if not color_models:
        create_models(clusters)
    # else:
        # Get voxel colors based on K means clustering and color model matching
        # colors = set_voxel_colors(clusters, parse_frame)

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
    # the number of cluster's (3 per person; R,G,B or H,S,V)
    N = 3

    #create + train EM model
    em = cv.ml.EM_create()
    em.setClustersNumber(N)
    ret, logLikelihoods, labels, probs = em.trainEM(person_color)

    #predict:
        #output[1]: contains as many columns as you have clusters (3 per person, RGB).
        #Each value is the likelihood (probability) of observing that cluster.
        #Read as: if this pixel came from this person, what would be the probability that it belongs to each of the clusters.

    predict_labels = em.predict(person_color)

    #get mean prediction values (sum of all three channels = 1)
    predict_mean = predict_labels[1].mean(axis=0)


    return logLikelihoods, labels, probs, predict_labels, predict_mean

'''
Function that creates a color model out of one frame given 3D voxel coordinates.
'''
def create_models(clusters):
    global color_models

    frame = cv.imread('images/frame.jpg')
    frame_online = cv.imread('images/frame2.jpg')
    copy = np.copy(frame)

    #color_models = voxels_hsv(clusters, frame)
    person_1_off, person_2_off, person_3_off, person_4_off = voxels_hsv(clusters, frame)
    print(f'person_1_off:{person_1_off}')

    person_1_on, person_2_on, person_3_on, person_4_on = voxels_hsv(clusters, frame_online)

    #create + train EM offline model for all persons
    logLikelihoods1, labels1, probs1, predict_labels1, predict_mean1 = EM_models(person_1_off)
    logLikelihoods2, labels2, probs2, predict_labels2, predict_mean2 = EM_models(person_2_off)
    logLikelihoods3, labels3, probs3, predict_labels3, predict_mean3 = EM_models(person_3_off)
    logLikelihoods4, labels4, probs4, predict_labels4, predict_mean4 = EM_models(person_4_off)

    #print(f'logLikelihoods1:{logLikelihoods1[0:10]}')
    #print(f'labels1:{labels1[0:10]}')
    #print(f'probs1:{probs1[0:10]}')

    #print(f'logLikelihoods2:{logLikelihoods2[0:10]}')
    #print(f'labels2:{labels2[0:10]}')
    #print(f'probs2:{probs2[0:10]}')


    #print(f'len(person_1_RGB):{len(person_1_RGB)}')
    print(f'predict_label1: {predict_labels1}')
    #print(f'len predict_label1: {len(predict_labels1[1])}')
    #print(f'type predict_label1: {type(predict_labels1[1])}')
    
    #print(f'predict_labels2: {predict_labels2}')
    #print(f'predict_labels3: {predict_labels3}')
    #print(f'predict_labels4: {predict_labels4}')

    print(f'pred_mean1:{predict_mean1}')
    print(f'pred_mean2:{predict_mean2}')
    print(f'pred_mean3:{predict_mean3}')
    print(f'pred_mean4:{predict_mean4}')

    #Online

    #TO DO:
    #stap 1: maak heel basaal online model af
        # - matching: which cluster with which person


    logLikelihoods1_on, labels1_on, probs1_on, predict_labels1_on, predict_mean1_on = EM_models(person_1_on)
    logLikelihoods2_on, labels2_on, probs2_on, predict_labels2_on, predict_mean2_on = EM_models(person_2_on)
    logLikelihoods3_on, labels3_on, probs3_on, predict_labels3_on, predict_mean3_on = EM_models(person_3_on)
    logLikelihoods4_on, labels4_on, probs4_on, predict_labels4_on, predict_mean4_on = EM_models(person_4_on)

    print(f'predict_label1_on: {predict_labels1_on}')
    #print(f'len predict_label1: {len(predict_labels1[1])}')
    #print(f'type predict_label1: {type(predict_labels1[1])}')
    
    #print(f'predict_labels2: {predict_labels2}')
    #print(f'predict_labels3: {predict_labels3}')
    #print(f'predict_labels4: {predict_labels4}')

    print(f'predict_mean1_on:{predict_mean1_on}')
    print(f'predict_mean2_on:{predict_mean2_on}')
    print(f'predict_mean3_on:{predict_mean3_on}')
    print(f'predict_mean4_on:{predict_mean4_on}')

    #distance voor person 1 to all 4 models

    dist1_1 = [np.linalg.norm(predict_mean1[i]-predict_mean1_on[i]) for i in range(len(predict_mean1_on))]
    dist1_2 = [np.linalg.norm(predict_mean1[i]-predict_mean2_on[i]) for i in range(len(predict_mean1_on))]
    dist1_3 = [np.linalg.norm(predict_mean1[i]-predict_mean3_on[i]) for i in range(len(predict_mean1_on))]
    dist1_4 = [np.linalg.norm(predict_mean1[i]-predict_mean4_on[i]) for i in range(len(predict_mean1_on))]

    print(f'dist1_1:{dist1_1}')
    print(f'dist1_2:{dist1_2}')
    print(f'dist1_3:{dist1_3}')
    print(f'dist1_4:{dist1_4}')


    #dist_R = np.linalg.norm(predict_mean1[0]-predict_mean1_on[0])
    #dist_B = np.linalg.norm(predict_mean1[1]-predict_mean1_on[1])
    #dist_G = np.linalg.norm(predict_mean1[2]-predict_mean1_on[2])

    #print(f'dist_R: {dist_R}')
    #print(f'dist_B: {dist_B}')
    #print(f'dist_G: {dist_G}')


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

    #show person 1 mask


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

  
    #save color values of voxels corresponding to persons
    person_1_RGB = np.float32(RGB_list[0])
    person_2_RGB = np.float32(RGB_list[1])
    person_3_RGB = np.float32(RGB_list[2])
    person_4_RGB = np.float32(RGB_list[3])

    #print(f'RGB_list 1:{person_1_RGB}')
    #print(f'RGB_list 2:{person_2_RGB}')
    #print(f'RGB_list 3:{person_3_RGB}')
    #print(f'RGB_list 4:{person_4_RGB}')

    return person_1_RGB, person_2_RGB, person_3_RGB, person_4_RGB







                   

    
    

'''
Compare each cluster to the created color models to determine which cluster
belongs to which person.
'''
def set_voxel_colors(clusters, parse_frame):
    hsv_list = voxels_hsv(clusters, parse_frame)

    '''
    Compare the hsv histogram of a new frame to each color model (of each person)
    and decide which cluster should get which color (based on distance/similarity).
    Hopefully, we end up with a list of colors that can be passed back.
    '''

    # return colors
    return None


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
