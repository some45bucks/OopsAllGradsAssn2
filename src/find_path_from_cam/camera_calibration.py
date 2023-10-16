import numpy as np
import cv2 as cv
import glob
import os

def calibrate_camera(x, y):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
    objpoints = []  # 3D point in real-world space
    imgpoints = []  # 2D points in the image plane

    directory_path = './data/camera_calibration/calibration_images'
    full_directory_path = os.path.join(os.getcwd(), directory_path)

    images = glob.glob(os.path.join(full_directory_path, '**/*.jpg'), recursive=True)
    gray = None
    corners = np.array([])

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (x, y), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, images 

def undistort_images(mtx, dist, images):
    undistorted_image_dir = './data/camera_calibration/undistorted_images'
    os.makedirs(undistorted_image_dir, exist_ok=True)

    for fname in images:
        img = cv.imread(fname)
        undistorted_image = cv.undistort(img, mtx, dist)

        # Create an image highlighting the differences
        difference_image = cv.absdiff(img, undistorted_image)

        image_name = os.path.basename(fname)
        concatenated_image_path = os.path.join(undistorted_image_dir, 'concatenated_' + image_name)
        difference_image_path = os.path.join(undistorted_image_dir, 'difference_' + image_name)

        # Concatenate the original and undistorted images side by side
        concatenated_image = np.hstack((img, undistorted_image))

        cv.imwrite(concatenated_image_path, concatenated_image)
        cv.imwrite(difference_image_path, difference_image)

def save_calibration_data(mtx, dist):
    with open('./data/camera_calibration/calibration_data.txt', 'w') as f:
        for i in mtx:
            for j in i:
                f.write(f'{j} ')
            f.write('\n')
        for i in dist:
            for j in i:
                f.write(f'{j} ')
            f.write('\n')

mtx, dist, images = calibrate_camera(6, 8)  
save_calibration_data(mtx, dist)
undistort_images(mtx, dist, images)
