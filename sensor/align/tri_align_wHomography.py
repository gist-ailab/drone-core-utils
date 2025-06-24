import glob
import numpy as np
import cv2
import pickle
import os


ROOT = '/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/ros_bag_api/out_aligned'
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


depth_folder_name = 'ilidar_depth_aligned'
intensity_folder_name = 'ilidar_intensity_aligned'
rgb_folder_name = 'rgb_aligned'
thermal_folder_name = 'ir_aligned'

# check all folders with all modality

available_folder_list = []
folders = os.listdir(ROOT)
for folder in folders:
    scene_folder_path = os.path.join(ROOT, folder)
    subfolders = os.listdir(scene_folder_path)
    if 'ilidar_depth' in subfolders and 'ilidar_intensity' in subfolders and 'ir_image_raw' in subfolders and 'rgb_image_raw' in subfolders:
        available_folder_list.append(scene_folder_path)


modality_name_list = ['ilidar_depth', 'ilidar_intensity', 'rgb_image_raw', 'ir_image_raw']
h, w = 640, 480

for subfolder in available_folder_list:
    aligned_depth_folder = os.path.join(subfolder, depth_folder_name)
    aligned_intentisy_folder= os.path.join(subfolder, intensity_folder_name)
    aligned_rgb_folder = os.path.join(subfolder, rgb_folder_name)
    aligned_thermal_folder = os.path.join(subfolder, thermal_folder_name)
    os.makedirs(aligned_depth_folder, exist_ok=True)
    os.makedirs(aligned_intentisy_folder, exist_ok=True)
    os.makedirs(aligned_rgb_folder, exist_ok=True)
    os.makedirs(aligned_thermal_folder, exist_ok = True)

    depth_source_folder= os.path.join(subfolder, modality_name_list[0])
    depth_source_files = os.listdir(depth_source_folder)
    for depth_source_file in depth_source_files:
        depth_img = cv2.imread(os.path.join(depth_source_folder, depth_source_file))
        depth_rotated = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
        depth_undistorted = cv2.undistort(depth_rotated, lidar_mtx, lidar_dist)
        aligned_depth = cv2.warpPerspective(depth_undistorted, H_lidar2rgb, (w, h))
        cv2.imwrite(f'{aligned_depth_folder}/{depth_source_file}', aligned_depth)
        print(f'Aligned depth saved | {aligned_depth_folder}/{depth_source_file}')

    intensity_source_folder= os.path.join(subfolder, modality_name_list[1])
    intensity_source_files = os.listdir(intensity_source_folder)
    for intensity_source_file in intensity_source_files:
        intensity_img = cv2.imread(os.path.join(intensity_source_folder, intensity_source_file))
        intensity_rotated = cv2.rotate(intensity_img, cv2.ROTATE_90_CLOCKWISE)
        intensity_undistorted = cv2.undistort(intensity_rotated, lidar_mtx, lidar_dist)
        aligned_intensity = cv2.warpPerspective(intensity_undistorted, H_lidar2rgb, (w, h))
        cv2.imwrite(f'{aligned_intentisy_folder}/{intensity_source_file}', aligned_intensity)
        print(f'Aligned intensity saved | {aligned_intentisy_folder}/{intensity_source_file}')   

    
    rgb_source_folder= os.path.join(subfolder, modality_name_list[2])
    rgb_source_files = os.listdir(rgb_source_folder)
    for rgb_source_file in rgb_source_files:
        rgb_img = cv2.imread(os.path.join(rgb_source_folder, rgb_source_file))
        rgb_rotated = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(f'{aligned_rgb_folder}/{rgb_source_file}', rgb_rotated)
        print(f'Aligned rgb saved | {aligned_rgb_folder}/{rgb_source_file}')    


    thermal_source_folder= os.path.join(subfolder, modality_name_list[3])
    thermal_source_files = os.listdir(thermal_source_folder)
    for thermal_source_file in thermal_source_files:
        thermal_img = cv2.imread(os.path.join(thermal_source_folder, thermal_source_file))
        thermal_img = cv2.resize(thermal_img, (640,480))
        thermal_undistorted = cv2.undistort(thermal_img, thermal_mtx, thermal_dist)
        aligned_thermal = cv2.warpPerspective(thermal_undistorted, H_thermal2rgb, (w, h))
        cv2.imwrite(f'{aligned_thermal_folder}/{thermal_source_file}', aligned_thermal)
        print(f'Aligned thermal saved | {aligned_thermal_folder}/{thermal_source_file}')   

    


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