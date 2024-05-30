def set_voxel_positions(vc, path, path_colors, models):
    # Gets voxel locations from voxel reconstructor
    voxels, colors, parse_frame = vc.next_frame()

    ############################################################################
    '''
    Translate all voxels in voxel volume by setting y coordinate to 0 (height).
    Use these coordinates to find 4 clusters using K means. For visualisation purposes,
    add the found centers to the voxel data, along with four distinct colors.
    '''

    body_voxels = [voxel for voxel in voxels if voxel[1] > 12 and voxel[1] < 26]

    # Translate voxel data
    temp = [[voxel[0], 0, voxel[2]] for voxel in body_voxels]

    # Find clusters
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 4
    _, labels, centers = cv.kmeans(np.float32(temp), k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    '''
    Compare with color model
    '''

    # Translate voxel data
    hsv_list = voxels_hsv(parse_frame, body_voxels, labels)
    color_model = []
    for model in models:
        for hsv in range(len(hsv_list)):
            continue
            # kijk teams vraag van rick otter en ronald




    '''
    Create path

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


    '''
    Based on previous center
    '''

    # if len(prev_centers) != 0:
    #     colors = []
    #     for x in range(len(centers)): #loop over alle nieuwe centers
    #         smallest_dst = 999
    #         best = -1
    #         for y in range(len(prev_centers)): #loop over alle oude centers
    #             squared_dist = np.sum((centers[x]-prev_centers[y])**2, axis=0)
    #             dist = np.sqrt(squared_dist)
    #             if dist < smallest_dst: #selecteer kleinste distance tussen een center
    #                 smallest_dst = dist
    #                 best = y
    #         colors.append(prev_colors[best]) #en pak de kleur index van de beste prev_center die matcht bij de nieuwe center

    #     # geef door voor volgende run
    #     prev_centers = centers
    #     prev_colors = colors
    # else: # eerste keer, gebruik de kleuren volgorde waarmee geinitialiseerd
    #     prev_centers = centers
    #     colors = prev_colors

    # label ze opnieuw op basis van de kleuren volgorde geselecteerd met de distance functie
    # colors_data = [[0,0,0]]*len(data)
    # for i in range(len(data)):
    #     if labels[i][0] == 0:
    #         colors_data[i] = colors[0]
    #     elif labels[i][0] == 1:
    #         colors_data[i] = colors[1]
    #     elif labels[i][0] == 2:
    #         colors_data[i] = colors[2]
    #     else:
    #         colors_data[i] = colors[3]

    # colors_data = [[0,0,0]]*len(data)
    # for i in range(len(data)):
    #     if labels[i][0] == 0:
    #         colors_data[i] = colors[0]
    #     elif labels[i][0] == 1:
    #         colors_data[i] = colors[1]
    #     elif labels[i][0] == 2:
    #         colors_data[i] = colors[2]
    #     else:
    #         colors_data[i] = colors[3]

    return voxels, colors, path, path_colors



def voxels_hsv(frame, voxels, labels):

    # no pants no head
    # voxels = [voxel for voxel in voxels if voxel[1] > 12 and voxel[1] < 26]

    # Divide clusters over unique lists and concatenate
    person_1 = [voxels[i] for i in range(len(voxels)) if labels[i] == 0]
    person_2 = [voxels[i] for i in range(len(voxels)) if labels[i] == 1]
    person_3 = [voxels[i] for i in range(len(voxels)) if labels[i] == 2]
    person_4 = [voxels[i] for i in range(len(voxels)) if labels[i] == 3]
    persons = [person_1, person_2, person_3, person_4]

    '''
    Get 2D coordinates by projecting the 3D voxel coordinates (werkte nog ff niet, en is miss trager)
    '''
    # cam2 = VoxelCam(2)
    # mtx, dist, rvec, tvec, _ = cam2.get_info().values()

    # imgpts = cv.projectPoints(np.float32(voxels), rvec, tvec, mtx, dist)[0]

    # Project the clustered 3D voxels to the 2D image plane of Cam2
    # imgpts_1 = [cv.projectPoints(np.float32(voxel), rvec, tvec, mtx, dist)[0] for voxel in person_1]
    # imgpts_2 = [cv.projectPoints(np.float32(voxel), rvec, tvec, mtx, dist)[0] for voxel in person_2]
    # imgpts_3 = [cv.projectPoints(np.float32(voxel), rvec, tvec, mtx, dist)[0] for voxel in person_3]
    # imgpts_4 = [cv.projectPoints(np.float32(voxel), rvec, tvec, mtx, dist)[0] for voxel in person_4]
    # imgpts = [imgpts_1, imgpts_2, imgpts_3, imgpts_4]

    # # Visualise the retrieved 2D image points by drawing them on the original frame
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    # for i in range(4):
    #     for pt in imgpts[i]:
    #         cv.circle(frame, (int(pt[0][0][0]), int(pt[0][0][1])), 2, colors[i], thickness=cv.FILLED)

    '''
    Find 2D coordinates by using the original look-up table
    '''
    table_path = 'data/table/cam2.npz'
    with np.load(table_path, allow_pickle=True) as f_table:
        table = f_table['table'].tolist()

    # Retrieve the 2D coordinates (image points) that correspond to the 3D voxels (for each cluster)
    imgpts = []
    for person in persons:
        pts = []
        for pt in table.keys():
            voxel = list(table[pt][0])
            voxel = [voxel[0]-(int(RESOLUTION)/2), -voxel[2]+RESOLUTION, voxel[1]-(int(RESOLUTION)/2)]
            if voxel in person:
                pts.append(pt)
        imgpts.append(pts)






hsv_list = voxels_hsv(copy, voxels, labels)

models = []
for p in hsv_list:
    continue
    # model = mixture.GaussianMixture(covariance_type='diag')
    # model.fit(p)
    # models.append(model)
    # print(model.predict_proba(hsv_list[0]))
    # visualize_3d_gmm(np.array(p), model.weights_, model.means_.T, np.sqrt(model.covariances_).T)

return models
