import glob
import numpy as np
import os
import cv2

root = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250501_4'

depth_list = glob.glob(os.path.join(root, 'lidar_hybo', '*_raw_depth.npy'))
depth_list.sort()
target_w = 480
target_h = 640


for depth_file in depth_list:
    raw_depth = np.load(depth_file)
    raw_intensity = np.load(depth_file.replace('_raw_depth.npy', '_raw_intensity.npy'))
    raw_intensity_img = cv2.imread(depth_file.replace('_raw_depth.npy', '_raw_intensity.png'), cv2.IMREAD_UNCHANGED)
    # Resize using nearest neighbor interpolation
    resized_depth = cv2.resize(raw_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    resized_intensity = cv2.resize(raw_intensity, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    resized_intensity_img = cv2.resize(raw_intensity_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # Save resized files
    np.save(depth_file, resized_depth)
    np.save(depth_file.replace('_raw_depth.npy', '_raw_intensity.npy'), resized_intensity)
    cv2.imwrite(depth_file.replace('_raw_depth.npy', '_raw_intensity.png'), resized_intensity_img)

    print(f"Processed and saved resized files for {depth_file}")