from exercise_1 import box3d, projectpoints
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


def radial_distort(p, dist):
    """
    Distorts the points in vector P, based on distortion coefficients dist.
        rd = r(1+dr)
        dr = k3||r||^2 + k5||r||^2 + ...
    
    dist holds the k3, k5, ... koefficients and p holds all all vectors.
    """
    dist_p = np.zeros(shape=(p.T).shape)
    i = 0
    for r in p.T:
        dr = 0
        j = 2
        for k in dist:
            dr += k*np.linalg.norm(r)**j
            j+=2

        dist_p[i] = r*(1+dr)
        i+=1
    return dist_p.T


def projectpoints_2(K, R, t, Q, dist):
    """
    Projects 3D points of an object to the 2D image plane of the camera.
    Using homogenous coordinates the process can be done like this:
            p_h = K*[R t]*P_h

    Parameters:
        - K: camera matrix - hold intrinsic camera info like focault distance and principal points
        - R, t: pose of camera transformation; scale and transport object to the camera plane.
        - Q: 3xn the n 3D points to be projected onto image plane.
        - dist: distortion coefficients list.
    
    Returns: 2xn matrix of projected points
    """
    # First creates the [R t] Matrix
    A = np.hstack((R, t))
    # Then, translate Q to homogenous plane => 4xn matrix by adding s=1
    B = np.vstack((Q, np.ones(len(Q[0]))))
    C = A@B
    # Return to non-homogenous coordinates
    p = C[0:2, :]/C[2, :]
    # Distort the result
    p = radial_distort(p, dist)
    q = np.vstack((p, np.ones(len(p[0]))))
    # Multiply by the camera matrix
    q = K@q
    # Translate back to cartesian coordinates and return (divide all by s, then remove s)
    return q[0:2, :]/q[2, :]

if __name__ == "__main__":
    
    # 2.1 Skew
    Q = box3d(16)
    f = 600
    alpha = 1
    beta = 0
    dx = 400
    dy = 400
    K = np.array([[f, f*beta, dx], [0, f*alpha, dy], [0, 0, 1]])
    t = np.array([[0, 0.2, 1.5]]).T
    R = np.identity(3)
    p = projectpoints(K, R, t, Q)
    plt.scatter(p[0,:], p[1,:])
    plt.show()

    # Where does the corner P_1 = [−0.5, −0.5, −0.5] project to? 
    P_1 = np.asfarray([[-0.5, -0.5, -0.5]]).T
    p = projectpoints(K, R, t, P_1)
    #print(p) # [100, 220].T

    # Does the object fit the image?
    # => dx, dy = 400 => 800x800 image, so some values are outside now.
    
    # 2.2 Adding distortion
    dist = [-0.2]
    p = projectpoints_2(K, R, t, Q, dist)
    plt.scatter(p[0,:], p[1,:])
    plt.show()

    # Where does the corner P_1 = [−0.5, −0.5, −0.5] project to? 
    P_1 = np.asfarray([[-0.5, -0.5, -0.5]]).T
    p = projectpoints_2(K, R, t, P_1, dist)
    print(p) # [120.4, 232.24].T

    # Now the image will entirely be captured do to negative distortion coefficient.

    #