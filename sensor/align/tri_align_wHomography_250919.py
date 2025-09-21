import glob
import numpy as np
import cv2
import pickle
import os
import glob

ROOT = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/DATA/synced/out_sync'
GROUP_PARENTDIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco/images/group_rgb'
group_rgb_lists = glob.glob(os.path.join(GROUP_PARENTDIR, '**', '*.png'), recursive=True)

# 저장할 모달리티별 부모 디렉토리 설정
SAVE_BASE_DIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco_synced/images'
GROUP_DEPTH_DIR = os.path.join(SAVE_BASE_DIR, 'group_depth')
GROUP_INTENSITY_DIR = os.path.join(SAVE_BASE_DIR, 'group_intensity')  
GROUP_IR_DIR = os.path.join(SAVE_BASE_DIR, 'group_ir')
GROUP_RGB_DIR = os.path.join(SAVE_BASE_DIR, 'group_rgb')
GROUP_RGB_WOCALIB_DIR = os.path.join(SAVE_BASE_DIR, 'group_rgb_wocalib')

def find_matching_rgb_file(source_filename, group_rgb_lists):
    """
    source_filename에서 frame_ prefix를 추가하여 group_rgb_lists에서 매칭되는 파일을 찾는 함수
    """
    target_filename = 'frame_' + source_filename
    
    for rgb_path in group_rgb_lists:
        if os.path.basename(rgb_path) == target_filename:
            return rgb_path
    return None

def get_group_folder_from_rgb_path(rgb_path):
    """
    RGB 파일 경로에서 그룹 폴더명(예: group_00)을 추출하는 함수
    """
    path_parts = rgb_path.split(os.sep)
    for part in path_parts:
        if part.startswith('group_') and part != 'group_rgb':
            return part
    return None

def create_save_path(modality_base_dir, group_folder, filename):
    """
    모달리티별 저장 경로를 생성하고 디렉토리를 만드는 함수
    """
    save_dir = os.path.join(modality_base_dir, group_folder)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


lidar_pickle = '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1/lidar_hybo_calibration/lidar_calibration.pkl'
thermal_pickle = '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/meta/thermal_calibration.pkl'
rgb_pickle = '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/meta/rgb_calibration.pkl'
H_lidar2rgb_path = '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/H_lidar_hybo2rgb_arducam.npy'
H_thermal2rgb_path = '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/H_thermal_pure_resized_undistorted2rgb_arducam.npy'

with open(lidar_pickle, 'rb') as f:
    lidar_data = pickle.load(f)
with open(thermal_pickle, 'rb') as f:
    thermal_data = pickle.load(f)
with open(rgb_pickle, 'rb') as f:
    rgb_data = pickle.load(f)
lidar_mtx, lidar_dist = lidar_data['cam_dict']['mtx'], lidar_data['cam_dict']['dist']
thermal_mtx, thermal_dist = thermal_data['cam_dict']['mtx'], thermal_data['cam_dict']['dist']
rgb_mtx, rgb_dist = rgb_data['cam_dict']['mtx'], rgb_data['cam_dict']['dist']
H_lidar2rgb = np.load(H_lidar2rgb_path)
H_thermal2rgb = np.load(H_thermal2rgb_path)

print(f'Loaded homography matrices for LiDAR and IR to RGB \n \
LiDAR to RGB: \n{H_lidar2rgb} \n \
IR to RGB: \n{H_thermal2rgb}')

# check all folders with all modality
available_folder_list = []
folders = os.listdir(ROOT)
for folder in folders:
    scene_folder_path = os.path.join(ROOT, folder)
    subfolders = os.listdir(scene_folder_path)
    if 'ilidar_depth' in folder or 'ilidar_intensity' in folder or 'ir_image_raw' in folder or 'rgb_image_raw' in folder:
        available_folder_list.append(scene_folder_path)

modality_name_list = ['ilidar_depth', 'ilidar_intensity', 'rgb_image_raw', 'ir_image_raw']
h, w = 640, 480


depth_source_files = glob.glob(os.path.join(ROOT, modality_name_list[0], '**', '*.png'), recursive=True)
intensity_source_files = glob.glob(os.path.join(ROOT, modality_name_list[1], '**', '*.png'), recursive=True)
rgb_source_files = glob.glob(os.path.join(ROOT, modality_name_list[2], '**', '*.png'), recursive=True)
thermal_source_files = glob.glob(os.path.join(ROOT, modality_name_list[3], '**', '*.png'), recursive=True)

depth_source_files.sort()
intensity_source_files.sort()
rgb_source_files.sort()
thermal_source_files.sort()

for depth_source_file in depth_source_files:
    depth_img = cv2.imread(depth_source_file)
    file_name = os.path.basename(depth_source_file)
    
    # RGB 파일에서 매칭되는 파일 찾기
    matching_rgb_path = find_matching_rgb_file(file_name, group_rgb_lists)
    if matching_rgb_path is None:
        print(f'Warning: No matching RGB file found for {file_name}')
        continue
        
    # 그룹 폴더 추출
    group_folder = get_group_folder_from_rgb_path(matching_rgb_path)
    if group_folder is None:
        print(f'Warning: Could not extract group folder from {matching_rgb_path}')
        continue
    
    # 저장 경로 생성
    save_path = create_save_path(GROUP_DEPTH_DIR, group_folder, 'frame_' + file_name)
    
    depth_rotated = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
    depth_undistorted = cv2.undistort(depth_rotated, lidar_mtx, lidar_dist)
    aligned_depth = cv2.warpPerspective(depth_undistorted, H_lidar2rgb, (w, h))
    
    cv2.imwrite(save_path, aligned_depth)
    print(f'Aligned depth saved | {save_path}')

for intensity_source_file in intensity_source_files:
    intensity_img = cv2.imread(intensity_source_file)
    file_name = os.path.basename(intensity_source_file)
    
    # RGB 파일에서 매칭되는 파일 찾기
    matching_rgb_path = find_matching_rgb_file(file_name, group_rgb_lists)
    if matching_rgb_path is None:
        print(f'Warning: No matching RGB file found for {file_name}')
        continue
        
    # 그룹 폴더 추출
    group_folder = get_group_folder_from_rgb_path(matching_rgb_path)
    if group_folder is None:
        print(f'Warning: Could not extract group folder from {matching_rgb_path}')
        continue
    
    # 저장 경로 생성
    save_path = create_save_path(GROUP_INTENSITY_DIR, group_folder, 'frame_' + file_name)
    
    intensity_rotated = cv2.rotate(intensity_img, cv2.ROTATE_90_CLOCKWISE)
    intensity_undistorted = cv2.undistort(intensity_rotated, lidar_mtx, lidar_dist)
    aligned_intensity = cv2.warpPerspective(intensity_undistorted, H_lidar2rgb, (w, h))
    
    cv2.imwrite(save_path, aligned_intensity)
    print(f'Aligned intensity saved | {save_path}')   



for rgb_source_file in rgb_source_files:
    rgb_img = cv2.imread(rgb_source_file)
    file_name = os.path.basename(rgb_source_file)
    
    # RGB 파일에서 매칭되는 파일 찾기
    matching_rgb_path = find_matching_rgb_file(file_name, group_rgb_lists)
    if matching_rgb_path is None:
        print(f'Warning: No matching RGB file found for {file_name}')
        continue
        
    # 그룹 폴더 추출
    group_folder = get_group_folder_from_rgb_path(matching_rgb_path)
    if group_folder is None:
        print(f'Warning: Could not extract group folder from {matching_rgb_path}')
        continue
    
    # 이미지 처리
    rgb_rotated = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb_undistorted = cv2.undistort(rgb_rotated, rgb_mtx, rgb_dist)
    
    # 1. undistort 적용된 버전 저장 (group_rgb)
    save_path_undistorted = create_save_path(GROUP_RGB_DIR, group_folder, 'frame_' + file_name)
    cv2.imwrite(save_path_undistorted, rgb_undistorted)
    print(f'RGB undistorted saved | {save_path_undistorted}')
    
    # 2. undistort 적용되지 않은 버전 저장 (group_rgb_wocalib) - rotate만 적용
    save_path_wocalib = create_save_path(GROUP_RGB_WOCALIB_DIR, group_folder, 'frame_' + file_name)
    cv2.imwrite(save_path_wocalib, rgb_rotated)
    print(f'RGB w/o calib saved | {save_path_wocalib}')    


for thermal_source_file in thermal_source_files:
    thermal_img = cv2.imread(thermal_source_file)
    file_name = os.path.basename(thermal_source_file)
    
    # RGB 파일에서 매칭되는 파일 찾기
    matching_rgb_path = find_matching_rgb_file(file_name, group_rgb_lists)
    if matching_rgb_path is None:
        print(f'Warning: No matching RGB file found for {file_name}')
        continue
        
    # 그룹 폴더 추출
    group_folder = get_group_folder_from_rgb_path(matching_rgb_path)
    if group_folder is None:
        print(f'Warning: Could not extract group folder from {matching_rgb_path}')
        continue
    
    # 저장 경로 생성
    save_path = create_save_path(GROUP_IR_DIR, group_folder, 'frame_' + file_name)
    
    thermal_img = cv2.resize(thermal_img, (640,480))
    thermal_undistorted = cv2.undistort(thermal_img, thermal_mtx, thermal_dist)
    aligned_thermal = cv2.warpPerspective(thermal_undistorted, H_thermal2rgb, (w, h))
    cv2.imwrite(save_path, aligned_thermal)
    print(f'Aligned thermal saved | {save_path}')   

    
