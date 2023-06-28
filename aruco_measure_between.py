import cv2
import depthai as dai
import numpy as np
import pickle

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Properties
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

cam_rgb.preview.link(xout_rgb.input)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
aruco_length = 7.62 # cm => 3 in.

mtx = []
dist = []

with open('./calibration/rgb/calibration.pkl', 'rb') as f:
    cc = pickle.load(f)
    print(cc)
    mtx = cc[0]
    print(mtx)
    dist = cc[1]
    print(dist)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        preview = q.get()
        frame = preview.getCvFrame()

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        tvec_arr = []
        pts_arr = []

        if len(corners) > 0:
            for i in range(0, len(ids)):
                (top_left, top_right, bottom_right, bottom_left) = corners[i].reshape((4, 2))
                marker_corners = np.array([
                    top_left, 
                    top_right, 
                    bottom_right, 
                    bottom_left], 
                    dtype="float32"
                )

                objp = np.array([
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [1., 1., 0.],
                    [0., 1., 0.]], 
                    dtype="float32"
                )  
                axis = np.float32([[0., 0., 0.]])
                ret, rvecs, tvecs = cv2.solvePnP(objp, marker_corners, mtx, dist)
                center_point, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                center_point = np.int32(center_point).reshape(-1,2)
                pts_arr.append((int(top_left[0]),int(top_left[1])))

                if tvecs is not None:
                    rvec = np.squeeze(rvecs, axis=None)
                    tvec = np.squeeze(tvecs, axis=None)
                    tvec_arr.append(tvec)

        if len(tvec_arr) == 2:
            euc_dist = np.linalg.norm((tvec_arr[0] - tvec_arr[1]))*3
            pt1 = pts_arr[0]
            pt2 = pts_arr[1]
            cv2.line(frame, tuple(pts_arr[0]), tuple(pts_arr[1]), (0,255,0), 2)
            cv2.circle(frame, pt1, 2, (0,0,255), 2)
            cv2.circle(frame, pt2, 2, (0,0,255), 2)
            txtPt = (int((pt1[0] + pt2[0]) / 2), (int((pt1[1] + pt2[1]) / 2) - 10))
            cv2.putText(frame, f"{euc_dist:.2f} in", txtPt, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break