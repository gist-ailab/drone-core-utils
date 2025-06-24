import os
import glob
import numpy as np
import  cv2


lidar_path = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1/lidar_hybo"
# rgb_path = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/rgb_arducam"


def rotate_lidar(lidar_path):
    lidar_list = glob.glob(os.path.join(lidar_path, "*_raw_depth.npy"))
    lidar_list.sort()
    for lidar_file in lidar_list:
        # raw_depth = np.load(lidar_file)
        # raw_intensity = cv2.imread(lidar_file.replace('_raw_depth.npy', '_raw_intensity.png'), cv2.IMREAD_UNCHANGED)
        raw_intensity_np = np.load(lidar_file.replace('_raw_depth.npy', '_raw_intensity.npy'))


        # rotated_depth = np.rot90(raw_depth, k=1)  # 90도 시계 방향 회전
        # roated_intensity = cv2.rotate(raw_intensity, cv2.ROTATE_90_CLOCKWISE)  # 90도 시계 방향 회전
        rotated_intensity_np = np.rot90(raw_intensity_np, k=2)  # 90도 시계 방향 회전

        # np.save(lidar_file, rotated_depth)
        # cv2.imwrite(lidar_file.replace('_raw_depth.npy', '_raw_intensity.png'), roated_intensity)
        np.save(lidar_file.replace('_raw_depth.npy', '_raw_intensity.npy'), rotated_intensity_np)
        print(f"Rotated and saved: {lidar_file}")


def rotate_rgb(rgb_path):
    rgb_list = glob.glob(os.path.join(rgb_path, "*.png"))
    rgb_list.sort()
    for rgb_file in rgb_list:
        rgb_image = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
        rotated_rgb = cv2.rotate(rgb_image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90도 시계 방향 회전
        cv2.imwrite(rgb_file, rotated_rgb)
        print(f"Rotated and saved: {rgb_file}")
    
if __name__ == "__main__":
    rotate_lidar(lidar_path)
    # rotate_rgb(rgb_path)
    print("Rotation completed.")
