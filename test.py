import numpy as np
import cv2
import math

K=np.array([[1485.8200576708696, 0.0, 640.0], [0.0, 1505.6488713912988, 360.0], [0.0, 0.0, 1.0]])
D=np.array([[-0.28188393137315854], [2.512062461064215], [58.28060026590924], [-949.3977111921688]])

K_inv = np.linalg.inv(K)

print(K_inv)
def radiansToDegrees(theta):
    return theta * 180 / math.pi

def getAnglesFromPixels(K,D,pixel): #K matrix; distortion coefficients; pixel: (x,y). returns angles in degrees (t_x, t_y)

    (p_x, p_y) = (pixel[0], pixel[1])

    I = np.array([p_x, p_y, 1]) #image coordinates

    W = np.matmul(K_inv, I) #"world" coordinates (homogenous Z)

    # unused for now, not sure how effective it is
    # undistorted = cv2.undistortPoints(np.array([p_x,p_y], dtype=np.float64),K,D)[0][0]

    t_x = math.atan(W[0])
    t_y = -math.atan(W[1])

    return (radiansToDegrees(t_x),radiansToDegrees(t_y))

print(getAnglesFromPixels(K,D,(0,0)))
print(getAnglesFromPixels(K,D,(640,360)))

print(getAnglesFromPixels(K,D,(540,260)))