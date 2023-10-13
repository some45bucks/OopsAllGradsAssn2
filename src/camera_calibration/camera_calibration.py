import numpy as np
import cv2 as cv
import glob
import os
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

directory_path = './src/camera_calibration/calibration_images'
full_directory_path = os.path.join(os.getcwd(), directory_path)

images = glob.glob(os.path.join(full_directory_path, '**/*.jpg'), recursive=True)
gray = None
corners = np.array([])
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,7),corners=corners, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    print(corners)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

with open('./calibration_data.txt', 'w') as f:
    for i in mtx:
        for j in i:
            f.write(f'{j} \n')
    for i in dist:
        f.write(f'{i} \n')


