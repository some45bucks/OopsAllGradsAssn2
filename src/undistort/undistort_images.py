import cv2
import os
import numpy as np

input_folder = './images/'
output_folder = './undistorted_images/'


camera_matrix =  np.array(
        [[],
        [],
        []]
    )

distortion_coef =  []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    img = cv2.imread(os.path.join(input_folder, image_file))

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, dst)

    print(f"Undistorted and saved: {output_path}")

print("Undistortion of images completed.")
