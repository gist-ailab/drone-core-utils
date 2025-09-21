import glob
import numpy as np
import cv2
import pickle
import os
from typing import Dict, List

class ImageProcessor:
    """
    Multimodal image processor to handle calibration, alignment, and saving.
    """
    def __init__(self, root_dir: str, group_parent_dir: str, save_base_dir: str,
                 calib_files: Dict[str, str], homographies: Dict[str, str]):
        self.ROOT = root_dir
        self.GROUP_PARENTDIR = group_parent_dir
        self.SAVE_BASE_DIR = save_base_dir
        
        self.group_rgb_lists = glob.glob(os.path.join(self.GROUP_PARENTDIR, '**', '*.png'), recursive=True)
        self.group_rgb_map = {os.path.basename(p): p for p in self.group_rgb_lists}

        self.lidar_mtx, self.lidar_dist = self._load_calibration(calib_files['lidar'])
        self.thermal_mtx, self.thermal_dist = self._load_calibration(calib_files['thermal'])
        self.rgb_mtx, self.rgb_dist = self._load_calibration(calib_files['rgb'])
        self.H_lidar2rgb = np.load(homographies['lidar2rgb'])
        self.H_thermal2rgb = np.load(homographies['thermal2rgb'])
        
        # Corrected target dimensions based on user feedback: W=480, H=640
        self.w, self.h = 480, 640

    def _load_calibration(self, file_path: str):
        """Loads calibration data from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data['cam_dict']['mtx'], data['cam_dict']['dist']
        except FileNotFoundError:
            print(f"Error: Calibration file not found at {file_path}")
            return None, None
        
    def _find_matching_rgb_file(self, source_filename: str) -> str:
        """Finds matching RGB file path using a pre-built dictionary."""
        target_filename = 'frame_' + source_filename
        return self.group_rgb_map.get(target_filename, None)

    def _get_group_folder_from_rgb_path(self, rgb_path: str) -> str:
        """Extracts the group folder name from an RGB file path."""
        path_parts = rgb_path.split(os.sep)
        for part in path_parts:
            if part.startswith('group_') and part != 'group_rgb':
                return part
        return None

    def _create_save_path(self, modality_name: str, group_folder: str, filename: str) -> str:
        """Creates the save directory and returns the full path."""
        save_dir = os.path.join(self.SAVE_BASE_DIR, f'group_{modality_name}', group_folder)
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, filename)

    def process_and_save(self, modality_name: str, source_files: List[str]):
        """General method to process and save images for a given modality."""
        for source_file in source_files:
            try:
                img = cv2.imread(source_file)
                if img is None:
                    print(f"Warning: Could not read image at {source_file}")
                    continue

                file_name = os.path.basename(source_file)
                matching_rgb_path = self._find_matching_rgb_file(file_name)
                
                if not matching_rgb_path:
                    print(f'Warning: No matching RGB file found for {file_name}')
                    continue
                
                group_folder = self._get_group_folder_from_rgb_path(matching_rgb_path)
                if not group_folder:
                    print(f'Warning: Could not extract group folder from {matching_rgb_path}')
                    continue
                
                if modality_name in ['depth', 'intensity']:
                    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    img_undistorted = cv2.undistort(img_rotated, self.lidar_mtx, self.lidar_dist)
                    aligned_img = cv2.warpPerspective(img_undistorted, self.H_lidar2rgb, (self.w, self.h))
                    
                    save_path = self._create_save_path(modality_name, group_folder, 'frame_' + file_name)
                    cv2.imwrite(save_path, aligned_img)
                    print(f'Aligned {modality_name} saved | {save_path}')

                elif modality_name == 'rgb':
                    img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # 1. Save undistorted version
                    img_undistorted = cv2.undistort(img_rotated, self.rgb_mtx, self.rgb_dist)
                    save_path_undistorted = self._create_save_path('rgb', group_folder, 'frame_' + file_name)
                    cv2.imwrite(save_path_undistorted, img_undistorted)
                    print(f'RGB undistorted saved | {save_path_undistorted}')
                    
                    # 2. Save w/o calibration version (only rotated)
                    save_path_wocalib = self._create_save_path('rgb_wocalib', group_folder, 'frame_' + file_name)
                    cv2.imwrite(save_path_wocalib, img_rotated)
                    print(f'RGB w/o calib saved | {save_path_wocalib}')
                
                elif modality_name == 'ir':
                    # Restored to original logic: resize and then warpPerspective without undistort
                    thermal_img = cv2.resize(img, (640,480))
                    aligned_thermal = cv2.warpPerspective(thermal_img, self.H_thermal2rgb, (self.w, self.h))
                    
                    save_path = self._create_save_path('ir', group_folder, 'frame_' + file_name)
                    cv2.imwrite(save_path, aligned_thermal)
                    print(f'Aligned thermal saved | {save_path}')   
            
            except Exception as e:
                print(f"An error occurred while processing {source_file}: {e}")

def main():
    ROOT = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/DATA/synced/out_sync'
    GROUP_PARENTDIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco/images/group_rgb'
    SAVE_BASE_DIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco_synced/images'

    calib_files = {
        'lidar': '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1/lidar_hybo_calibration/lidar_calibration.pkl',
        'thermal': '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/meta/thermal_calibration.pkl',
        'rgb': '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/meta/rgb_calibration.pkl'
    }
    homographies = {
        'lidar2rgb': '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/H_lidar_hybo2rgb_arducam.npy',
        'thermal2rgb': '/media/ailab/HDD1/Workspace/src/Project/Drone24/sensors/H_thermal_pure_resized_undistorted2rgb_arducam.npy'
    }

    processor = ImageProcessor(ROOT, GROUP_PARENTDIR, SAVE_BASE_DIR, calib_files, homographies)

    modality_info = {
        'ir': 'ir_image_raw',
        'depth': 'ilidar_depth',
        'intensity': 'ilidar_intensity',
        'rgb': 'rgb_image_raw',
    }

    for modality, folder_name in modality_info.items():
        source_files = sorted(glob.glob(os.path.join(ROOT, folder_name, '**', '*.png'), recursive=True))
        processor.process_and_save(modality, source_files)

if __name__ == '__main__':
    main()