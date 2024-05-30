import glm
import numpy as np
import cv2 as cv
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def remove_outliers(centers, clusters):
    # Compute the distances of each voxel from the center
    dist_1 = [np.linalg.norm(i - centers[0], axis=0) for i in clusters[0]]
    dist_2 = [np.linalg.norm(i - centers[1], axis=0) for i in clusters[1]]
    dist_3 = [np.linalg.norm(i - centers[2], axis=0) for i in clusters[2]]
    dist_4 = [np.linalg.norm(i - centers[3], axis=0) for i in clusters[3]]
    distances1 = [dist_1, dist_2, dist_3, dist_4]

    # Calculate the median and standard deviation of distances
    median_distances = [np.median(i) for i in distances1]
    std_distances = [np.std(i) for i in distances1]

    # Define a threshold for outlier removal
    threshold_1 = median_distances[0] + std_distances[0]
    threshold_2 = median_distances[1] + std_distances[1]
    threshold_3 = median_distances[2] + std_distances[2]
    threshold_4 = median_distances[3] + std_distances[3]
    threshold_combo = [threshold_1, threshold_2, threshold_3,threshold_4]

    outlier_list = []
    for i in range(4):
        for j in range(len(clusters[i])):
            outliers = [voxel for voxel in clusters[i] if distances1[i][j] >= threshold_combo[i]]
        outlier_list.append(outliers)

    for i in range(4):
        for outlier in outlier_list[i]:
            clusters[i].remove(outlier)
     

    clusters_new = clusters[0] + clusters[1] +clusters[2] + clusters[3] 
    return clusters_new

def plot_kmeans(temp, labels, centers):

    #remove the y coordinate
    for t in temp:
        t.pop(1)

    #floate necessary for scatter plot
    temp = np.float32(temp)

    # Now separate the data, Note the flatten()
    A = temp[labels.ravel()==0]
    B = temp[labels.ravel()==1]
    C = temp[labels.ravel()==2]
    D = temp[labels.ravel()==3]

    # Plot the data
    plt.scatter(A[:,0],A[:,1], c = 'red')
    plt.scatter(B[:,0],B[:,1],c = 'blue')
    plt.scatter(C[:,0],C[:,1],c = 'green')
    plt.scatter(D[:,0],D[:,1],c = 'purple')
    plt.scatter(centers[:,0],centers[:,2],s = 80,c = 'yellow', marker = 's')
    plt.show()
    

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

    #remove outliers
    voxels_new = remove_outliers(centers, clusters)
    _, labels_new, centers_new = cv.kmeans(np.float32(voxels_new), k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # Divide clusters over unique lists and ignore voxels projected on the person's head or legs
    cluster_1 = [voxels_new[i] for i in range(len(voxels_new)) if labels_new[i][0] == 0 if (12 < voxels_new[i][1] < 27)]
    cluster_2 = [voxels_new[i] for i in range(len(voxels_new)) if labels_new[i][0] == 1 if (12 < voxels_new[i][1] < 27)]
    cluster_3 = [voxels_new[i] for i in range(len(voxels_new)) if labels_new[i][0] == 2 if (12 < voxels_new[i][1] < 27)]
    cluster_4 = [voxels_new[i] for i in range(len(voxels_new)) if labels_new[i][0] == 3 if (12 < voxels_new[i][1] < 27)]
    clusters = [cluster_1, cluster_2, cluster_3, cluster_4]

    #plot kmeans with and without outliers
    #plot_kmeans(temp, labels, centers)
    #plot_kmeans(voxels_new, labels_new, centers_new)


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

        print(f'temp_1: {temp_1}')
        print(f'colors_1: {colors_1}')

        temp_2 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 1]
        colors_2 = [options[color_labels[1]] for _ in range(len(temp_2))]

        temp_3 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 2]
        colors_3 = [options[color_labels[2]] for _ in range(len(temp_3))]

        temp_4 = [voxels[i] for i in range(len(voxels)) if labels[i][0] == 3]
        colors_4 = [options[color_labels[3]] for _ in range(len(temp_4))]

        # reorder colors based on model, then add to path using centers
        colors_p = np.array(options)[np.array(color_labels)].tolist()
        path.extend(centers_new.tolist())
        path_colors.extend(colors_p)

        # combine all
        voxels = temp_1 + temp_2 + temp_3 + temp_4 + path
        colors = colors_1 + colors_2 + colors_3 + colors_4 + path_colors

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

    frame = cv.imread('images/frame.jpg')
    copy = np.copy(frame)

    color_models = voxels_hsv(clusters, frame)


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

    # Create a list of rbg values of the 2D coordinates corresponding to the 3D voxels for each cluster
    cluster_frames = []
    # count = 1
    for pts in imgpts:
        print(len(pts))
        i, j = 0, 0
        size = int(np.sqrt(len(pts))) + 1
        cluster = np.zeros([size, size, 3], dtype=np.uint8)
        cluster.fill(255)
        for pt in pts:
            cluster[i][j] = frame[int(pt[0])][int(pt[1])]
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
        plt.plot(hist, color = col)
        plt.xlim([0, 254])
    return hist_list


'''
Compare the hsv histograms of a new frame to each color model (of each person)
and decide which cluster should get which color (based on distance/similarity).
'''
def set_voxel_colors(clusters, parse_frame):
    histograms = voxels_hsv(clusters, parse_frame)

    all_distances = []
    for hist in histograms:
        cluster_dist = []
        for model in color_models:
            curr_dist = 0
            for i in range(3):
                curr_dist += cv.compareHist(hist[i], model[i], cv.HISTCMP_CORREL)
            cluster_dist.append(curr_dist/3)
        all_distances.append(cluster_dist)

    label_1 = all_distances[0].index(max(all_distances[0]))
    label_2 = all_distances[1].index(max(all_distances[1]))
    label_3 = all_distances[2].index(max(all_distances[2]))
    label_4 = all_distances[3].index(max(all_distances[3]))
    labels = [label_1, label_2, label_3, label_4]

    #Hungarian matching
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
    # Create white image
    img = np.zeros([128,128,3],dtype=np.uint8)
    img.fill(255)

    # Add path coordinates
    for i in range(len(path)):
        # Transform 3D to 2D coordinates and translate to deal with negative numbers
        x = int(path[i][0]) + 50
        y = int(path[i][2]) + 50
        img[x][y] = colors[i]
    cv.imwrite('images/trace.jpg', img)
