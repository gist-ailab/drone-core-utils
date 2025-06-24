import cv2
import os
import glob

source_dir = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1/thermal_pure_new'
destination_dir = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1/thermal_pure_new_resized'
target_width = 640
target_height = 480

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all image files in the source directory
image_files = glob.glob(os.path.join(source_dir, '*.png')) # Assuming images are PNGs

for image_file in image_files:
    # Read the image
    img = cv2.imread(image_file)

    if img is None:
        print(f"Warning: Could not read image {image_file}. Skipping.")
        continue

    # Resize the image
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Get the base filename
    base_filename = os.path.basename(image_file)

    # Construct the full path for the output file
    output_path = os.path.join(destination_dir, base_filename)

    # Save the resized image
    cv2.imwrite(output_path, resized_img)
    print(f"Resized {base_filename} and saved to {output_path}")

print("Image resizing complete.")
