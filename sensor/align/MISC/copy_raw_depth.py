import shutil
import os
import glob



source_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250501_2/lidar_hybo'
target_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250501_3/lidar_hybo'



raw_depth_files = glob.glob(os.path.join(source_path, '*_raw_depth.npy'))
for file in raw_depth_files:
    filename = os.path.basename(file)
    target_file = os.path.join(target_path, filename)
    
    shutil.copy(file, target_file)
    print(f"Copied {file} to {target_file}")