import cv2
import os
import numpy as np

input_folder = './images/'
output_folder = './undistorted_images/'
calibration_path = './camera_calibration/calibration_data.txt'

def get_calibration_data(path):
    with open(path) as f:
        lines = f.readlines()
        vectors = []
        for i in range(3):
            vector = []
            for j in lines[i].split():
                vector.append(j)
            vectors.append(vector)
        camera_matrix = np.array(vectors)
        vector = []
        for i in lines[3].split():
            vector.append(i)
        distortion_coef = np.array(vector)
    return camera_matrix, distortion_coef

camera_matrix , distortion_coef = get_calibration_data(calibration_path)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    img = cv2.imread(os.path.join(input_folder, image_file))

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coef, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, distortion_coef, None, new_camera_matrix)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, dst)

    print(f"Undistorted and saved: {output_path}")

print("Undistortion of images completed.")
