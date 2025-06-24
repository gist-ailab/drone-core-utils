import glob
import os


root = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/thermal_pure_resized_undistorted'
corners = glob.glob(os.path.join(root, 'corners_*.png.npy'))


for corner in corners:
    dirname = os.path.dirname(corner)
    name = os.path.basename(corner)
    newname = name.replace('.png', "")
    filename = os.path.join(dirname, newname)
    os.rename(corner, filename)
    print(f"Renamed {corner} to {filename}")
    

