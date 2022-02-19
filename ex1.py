import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# Ex 1.10
def display_image(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    image = cv2.merge([r,g,b])
    plt.imshow(image)
    plt.show()

# 1.11
def box3d(n):
    """
    Creates a point-box in 3D with a cross in the middle.
    The box contains n points on each side, and along the cross.
    Useful for testing the camera model.
    """
    x = []
    y = []
    z = []

    for i, val in enumerate([-0.5, -0.5, 0, 0.5, 0.5], 1):
        x = np.concatenate((x, val*np.ones(n)))
        y = np.concatenate((y, np.linspace(-0.5, 0.5, n)))
        z = np.concatenate((z, (2*(i % 2)-1)*val*np.ones(n)))
    x2 = np.concatenate((x, y))
    x2 = np.concatenate((x2, x))
    y2 = np.concatenate((y, z))
    y2 = np.concatenate((y2, z))
    z2 = np.concatenate((z, x))
    z2 = np.concatenate((z2, y))
    # Vertical stack the vectors to a 3xn matrix Q
    return np.vstack((x2, y2, z2))

# EX 1.12 Projection
def projectpoints(K, R, t, Q):
    """
    Projects 3D points of an object to the 2D image plane of the camera.
    Using homogenous coordinates the process can be done like this:
            p_h = K*[R t]*P_h

    Parameters:
        - K: camera matrix - hold intrinsic camera info like focault distance and principal points
        - R, t: pose of camera transformation; scale and transport object to the camera plane.
        - Q: 3xn the n 3D points to be projected onto image plane.
    
    Returns: 2xn matrix of projected points
    """
    
    # First creates the [R t] Matrix
    A = np.hstack((R, t))
    # Then, translate Q to homogenous plane => 4xn matrix by adding s=1
    B = np.vstack((Q, np.ones(len(Q[0]))))
    # Solve the projection in homogenous plane 
    p_h = K@A@B
    # Translate back to cartesian coordinates and return (divide all by s, then remove s)
    return p_h[0:2, :]/p_h[2, :]

if __name__ == "__main__":
    '''
    # 1.10
    display_image("Castle.jpg")
    # 1.11 
    x, y, z = box3d(16)
    ax = plt.axes(projection='3d')  # syntax for 3-D projection
    ax.scatter3D(x, y, z)
    plt.show()
    '''
    # 1.12 
    Q = box3d(16)
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 0, 3]]).T
    R = K.copy()
    p = projectpoints(K, R, t, Q)
    print(Q.shape)
    plt.scatter(p[0,:], p[1,:])
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    plt.show()

    
    # 1.13 New R matrix (rotation)
    theta = math.radians(30)
    R = np.array([[math.cos(theta), 0, math.sin(theta)], [
             0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    p = projectpoints(K, R, t, Q)
    plt.scatter(p[0,:], p[1,:])
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    plt.show()
    
    # 1.14 what does R and t do?
    # R:
    # - Scales the object while transforming to camera plane
    # - Rotate, stretch or shrink the object
    # t: moves the object in the camera plane.
    
    # Rotate in other direction
    theta = math.radians(-30)
    R = np.array([[math.cos(theta), 0, math.sin(theta)], [
             0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    p = projectpoints(K, R, t, Q)
    plt.scatter(p[0,:], p[1,:])
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    plt.show()
    
    # Scale the object bigger
    theta = math.radians(-30)
    R = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]])
    p = projectpoints(K, R, t, Q)
    plt.scatter(p[0,:], p[1,:])
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    plt.show()

    # Stretch the object in y-axis
    theta = math.radians(-30)
    R = np.array([[0.5, 0, 0], [0, 1.5, 0], [0, 0, 0.5]])
    p = projectpoints(K, R, t, Q)
    plt.scatter(p[0,:], p[1,:])
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    plt.show()
