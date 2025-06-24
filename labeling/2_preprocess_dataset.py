'''
Post processing tools to prepare multi-modal dataset after downloading annotations and sources from supervisely
2025.06.24
maengjemo
'''
import os
import shutil
import glob

source_dir = "/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/ros_bag_api/labeling_250610"           #Original source folder taken from ROS
dest_dir = "/ailab_mat2/dataset/drone/drone_250610_dataset"                                                   #Folder to save the dataset        
image_dir = "/ailab_mat2/dataset/drone/drone_250610/images/train"                                             #Folder from supervisely

# Get a sorted list of subdirectories in the source directory
subfolders = sorted([
    f for f in os.listdir(source_dir)
    if os.path.isdir(os.path.join(source_dir, f)) and f.startswith("drone_")
])

group_num = 0
images = glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
for subfolder in subfolders:
    group_name = f"group_{group_num:02d}"
    group_path = os.path.join(dest_dir, group_name)
    os.makedirs(group_path, exist_ok=True)

    # Create subfolders inside the group folder
    rgb_aligned_path = os.path.join(group_path, "rgb_aligned")
    ilidar_depth_aligned_path = os.path.join(group_path, "ilidar_depth_aligned")
    ilidar_intensity_aligned_path = os.path.join(group_path, "ilidar_intensity_aligned")
    ir_aligned_path = os.path.join(group_path, "ir_aligned")

    os.makedirs(rgb_aligned_path, exist_ok=True)
    os.makedirs(ilidar_depth_aligned_path, exist_ok=True)
    os.makedirs(ilidar_intensity_aligned_path, exist_ok=True)
    os.makedirs(ir_aligned_path, exist_ok=True)

    # Copy files from source to destination
    src_labeling = os.path.join(source_dir, subfolder, "labeling_aligned")
    src_rgb_aligned = os.path.join(source_dir, subfolder, "rgb_aligned")
    src_ilidar_depth_aligned = os.path.join(source_dir, subfolder, "ilidar_depth_aligned")
    src_ilidar_intensity_aligned = os.path.join(source_dir, subfolder, "ilidar_intensity_aligned")
    src_ir_aligned = os.path.join(source_dir, subfolder, "ir_aligned")

    for file in glob.glob(os.path.join(src_labeling, "*")):
        filename = os.path.basename(file)
        shutil.copy(os.path.join(src_rgb_aligned, filename), os.path.join(rgb_aligned_path, filename))
        shutil.copy(os.path.join(src_ilidar_depth_aligned, filename), os.path.join(ilidar_depth_aligned_path, filename))
        shutil.copy(os.path.join(src_ilidar_intensity_aligned, filename), os.path.join(ilidar_intensity_aligned_path, filename))
        shutil.copy(os.path.join(src_ir_aligned, filename), os.path.join(ir_aligned_path, filename))

    try:
        last_digit = int(subfolder.split("_")[-1])
        if last_digit == 0:
            group_num += 1
    except ValueError:
        # Handle cases where the subfolder name doesn't end with a digit
        print(f"Warning: Could not extract last digit from subfolder name: {subfolder}")

print("Dataset preprocessing complete.")
