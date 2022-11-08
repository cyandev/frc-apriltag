import cv2
import math
import numpy as np
from pupil_apriltags import Detector

detector = Detector(families="tag36h11",debug=1)

def main():
    img = cv2.resize(cv2.imread("test/img2.JPG"), (1280,720))
    res = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # loop over the AprilTag detection results
    for r in res:
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
    cv2.imshow("testimg",img)
    cv2.waitKey(0)


def radiansToDegrees(theta):
    return theta * 180 / math.pi

def getAnglesFromPixels(K,D,pixel): #K matrix; distortion coefficients; pixel: (x,y). returns angles in degrees (t_x, t_y)

    (p_x, p_y) = (pixel[0], pixel[1])

    undistorted = cv2.undistortPoints(np.array([p_x,p_y], dtype=np.float64),K,D)[0][0]

    t_x = -(math.pi / 2 - math.atan2(1,undistorted[0]))
    t_y = -(math.pi / 2 - math.atan2(1,undistorted[1]))

    return (radiansToDegrees(t_x),radiansToDegrees(t_y))
main()