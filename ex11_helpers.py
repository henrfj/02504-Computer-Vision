import numpy as np
import cv2
import glob

def get_all_sift(gray):
    """
    Get stuff
    """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 1999) 
                                
    kp, desc = sift.detectAndCompute(gray, None) #Keypoints and descriptors

    point = np.array([k.pt for k in kp]).astype(np.int32)
    
    return point, kp, desc

def match_sift(des1, des2, KNN_matcher=False):
    """
    Match stuff
    """
    bf = cv2.BFMatcher() #crossCheck=True
    if KNN_matcher:
        knn_matches = bf.knnMatch(des1, des2, k=2)
        # Ratio test!
        ratio_matches = []
        for m,n in knn_matches:
            if m.distance < 0.75*n.distance:
                ratio_matches.append(m)
        
        # Return the keypoint indexes of matches 
        return np.array([(m.queryIdx, m.trainIdx) for m in ratio_matches])
    else:
        all_matches = bf.match(des1, des2)
        #sorted_matches = sorted(matches, key = lambda x:x.distance)
        return np.array([(m.queryIdx, m.trainIdx) for m in all_matches])
    

def first_three(K):
    """
    The first four exercises
    """
    ############
    ### 11.1 ###
    ############
    ext = ['png', 'jpg', 'gif'] # filetypes
    imdir = "Glyp/sequence/"
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGB2GRAY) for file in files]
    im0, im1, im2 = images[0], images[1], images[2]
    
    # POINTS
    point0, kp0, des0 = get_all_sift(im0)
    point1, kp1, des1 = get_all_sift(im1)
    point2, kp2, des2 = get_all_sift(im2)
    point0.shape, point1.shape, point2.shape
    # MATCHES
    matches01 = match_sift(des0, des1)
    matches12 = match_sift(des1, des2)
    matches01.shape, matches12.shape

    ############
    ### 11.2 ###
    ############

    # Essential RANSAC
    E01, Emask01 = cv2.findEssentialMat(point0[matches01[:, 0], :], point1[matches01[:, 1], :], K, method=cv2.RANSAC,
                            #prob=0.999, threshold=..., maxIters=...
                            )
    E12, Emask12 = cv2.findEssentialMat(point1[matches12[:, 0], :], point2[matches12[:, 1], :], K, method=cv2.RANSAC,
                            #prob=0.999, threshold=..., maxIters=...
                            )
    #_, R01, t01, mask01 = cv2.recoverPose(E01, point0, point1, K)
    #_, R12, t12, mask12 = cv2.recoverPose(E12, point1, point2, K)
    _, R01, t01, mask01 = cv2.recoverPose(E01, point0[matches01[:, 0], :], point1[matches01[:, 1], :], K)
    _, R12, t12, mask12 = cv2.recoverPose(E12, point1[matches12[:, 0], :], point2[matches12[:, 1], :], K)

    matches01_filtered = matches01[np.where(Emask01>0)[0]]
    matches12_filtered = matches12[np.where(Emask12>0)[0]]

    ############
    ### 11.3 ###
    ############    

    # Points shown in all three images
    _, idx01, idx12 = np.intersect1d(matches01_filtered[:,1], matches12_filtered[:,0], return_indices=True)

    # Extracting the valid matches - that go throughout the images
    validmatch01 = matches01_filtered[idx01, :]
    validmatch12 = matches12_filtered[idx12, :]

    # Extract the points belonging to the valid matches
    val0_idx = validmatch01[:, 0]
    val1_idx = validmatch01[:, 1]
    val2_idx = validmatch12[:, 1]
    # Transition points
    trans0 = point0[val0_idx, :]
    trans1 = point1[val1_idx, :]
    trans2 = point2[val2_idx, :]


    ###########
    ## 11. 4 ##
    ###########
    distCoeffs=np.zeros(5) 
    # Projection matrix using 
    P0 = K@np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K@np.hstack((R01, t01))
    #P2 = K@np.hstack((R12, t12))
    
    # Triangulate
    triang01 = cv2.triangulatePoints(P0, P1, trans0.T, trans1.T)    
    triang01 = triang01[:3]/triang01[3]

    return triang01


if __name__ == "__main__":
    K = np.loadtxt('Glyp/K.txt')
    triang01 =  first_three(K)
    print("Hello")
    print(triang01.shape)
    print("Hello")