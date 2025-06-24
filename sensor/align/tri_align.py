import glob
import numpy as np
import cv2
import pickle


lidar_pickle = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250525_1/lidar_hybo_calibration/calibration.pkl'
thermal_pickle = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/meta/thermal_calibration.pkl'

H_lidar2rgb_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/H_lidar_hybo2rgb_arducam.npy'
H_thermal2rgb_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/H_thermal_pure_resized_undistorted2rgb_arducam.npy'

with open(lidar_pickle, 'rb') as f:
    lidar_data = pickle.load(f)
with open(thermal_pickle, 'rb') as f:
    thermal_data = pickle.load(f)



lidar_mtx, lidar_dist = lidar_data['cam_dict']['mtx'], lidar_data['cam_dict']['dist']
thermal_mtx, thermal_dist = thermal_data['cam_dict']['mtx'], thermal_data['cam_dict']['dist']


H_lidar2rgb = np.load(H_lidar2rgb_path)
H_thermal2rgb = np.load(H_thermal2rgb_path)



rgb_img_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/rgb_arducam/005.png'
lidar_img_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/lidar_hybo/005_raw_intensity.png'
thermal_img_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/thermal_pure_resized_undistorted/005.png'

rgb_img = cv2.imread(rgb_img_path)
lidar_img = cv2.imread(lidar_img_path)
thermal_img = cv2.imread(thermal_img_path)

lidar_undistorted = cv2.undistort(lidar_img, lidar_mtx, lidar_dist)
h, w = rgb_img.shape[:2]

aligned_lidar = cv2.warpPerspective(lidar_undistorted, H_lidar2rgb, (w, h))
aligned_thermal = cv2.warpPerspective(thermal_img, H_thermal2rgb, (w, h))

thermal_resized = cv2.resize(thermal_img, (860, 640))
lidar_resized = cv2.resize(lidar_undistorted, (320, 640))
ov_lidar = cv2.addWeighted(rgb_img, 0.5, aligned_lidar, 0.5, -1)
ov_thermal = cv2.addWeighted(rgb_img, 0.5, aligned_thermal, 0.5, -1)
cat_lidar = cv2.hconcat([lidar_resized, rgb_img, aligned_lidar, ov_lidar])
cat_thermal = cv2.hconcat([thermal_resized, rgb_img, aligned_thermal, ov_thermal])

cv2.imwrite('1_tri_align_lidar.png', cat_lidar)
cv2.imwrite('1_tri_align_thermal.png', cat_thermal)