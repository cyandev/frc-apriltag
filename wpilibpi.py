import cscore
import json
import math
import numbers
import os
from threading import Thread
from networktables import NetworkTables
import numpy as np
import cv2

#camera calibration constants (fixme?)
K=np.array([[1485.8200576708696, 0.0, 640.0], [0.0, 1505.6488713912988, 360.0], [0.0, 0.0, 1.0]])
D=np.array([[-0.28188393137315854], [2.512062461064215], [58.28060026590924], [-949.3977111921688]])

K_inv = np.linalg.inv(K)

try:
    print("attempting to import apriltag...")
    import apriltag
    print("success!")
except:
    print("import failed, dowloading apriltag (THIS REQUIRES AN INTERNET CONNECTION)...")
    os.system(""" sudo date -s \"$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d\' \' -f5-8)Z\" """)
    os.system('pip3 install apriltag')
    import apriltag
    print("installation complete!")

def main():
    cs = cscore.CameraServer.getInstance()
    cs.enableLogging()

    camera = cscore.UsbCamera("Main",0)
    cvSink = cscore.CvSink("Video In")
    with open("./config.json","r") as f:
        print(json.dumps(json.load(f)))
        camera.setConfigJson("""{
    "height": 720,
    "width": 1280,
    "properties": [
        {
            "name": "connect_verbose",
            "value": 0
        },
        {
            "name": "brightness",
            "value": 33
        },
        {
            "name": "contrast",
            "value": 100
        },
        {
            "name": "saturation",
            "value": 50
        },
        {
            "name": "red_balance",
            "value": 1000
        },
        {
            "name": "blue_balance",
            "value": 1000
        },
        {
            "name": "horizontal_flip",
            "value": true
        },
        {
            "name": "vertical_flip",
            "value": true
        },
        {
            "name": "power_line_frequency",
            "value": 1
        },
        {
            "name": "sharpness",
            "value": 100
        },
        {
            "name": "color_effects",
            "value": 0
        },
        {
            "name": "rotate",
            "value": 0
        },
        {
            "name": "color_effects_cbcr",
            "value": 32896
        },
        {
            "name": "video_bitrate_mode",
            "value": 1
        },
        {
            "name": "video_bitrate",
            "value": 25000000
        },
        {
            "name": "repeat_sequence_header",
            "value": false
        },
        {
            "name": "h264_i_frame_period",
            "value": 60
        },
        {
            "name": "h264_level",
            "value": 11
        },
        {
            "name": "h264_profile",
            "value": 4
        },
        {
            "name": "auto_exposure",
            "value": 0
        },
        {
            "name": "exposure_time_absolute",
            "value": 9
        },
        {
            "name": "exposure_dynamic_framerate",
            "value": false
        },
        {
            "name": "auto_exposure_bias",
            "value": 12
        },
        {
            "name": "white_balance_auto_preset",
            "value": 1
        },
        {
            "name": "image_stabilization",
            "value": false
        },
        {
            "name": "iso_sensitivity",
            "value": 0
        },
        {
            "name": "iso_sensitivity_auto",
            "value": 1
        },
        {
            "name": "exposure_metering_mode",
            "value": 0
        },
        {
            "name": "scene_mode",
            "value": 0
        },
        {
            "name": "compression_quality",
            "value": 100
        }
    ]
}""")
    cvSink.setSource(camera)
    cvSinkThreaded = ThreadedCvSink(cvSink).start()

    outputStream = cs.putVideo("MainOut", 1280, 720)
    
    img = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

    print("starting main vision loop...")

    while True:
      # Tell the CvSink to grab a frame from the camera and put it
      # in the source image.  If there is an error notify the output.
        time, img = cvSinkThreaded.grabFrame()
        if time == 0: 
         # Send the output the error.
            outputStream.notifyError(cvSink.getError())
         # skip the rest of the current iteration
            continue
        res = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        if len(res) == 0:
            cv2.putText(img, "No Tags Found!", (0,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
        for r in res:
            (cA, cB, cC, cD) = r.corners
            (cX, cY) = (r.center[0], r.center[1])
            (tx, ty) = getAnglesFromPixels(K, D, (cX, cY))
            cv2.circle(img, (int(cX), int(cY)), 5, (255, 255, 0), -1)
            cv2.putText(img, str((round(tx,2), round(ty,2))), (int(cD[0]), int(cD[1])),cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,0), 2)

        outputStream.putFrame(img)


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

class ThreadedCvSink:
    def __init__(self,cvSink):
        self.cvSink = cvSink
        self.stopped = False
        self.img = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
        self.time = 0
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            self.time,self.img = self.cvSink.grabFrame(self.img)
    def grabFrame(self):
        return (self.time,self.img)
    def stop(self):
        self.stopped = True

main()