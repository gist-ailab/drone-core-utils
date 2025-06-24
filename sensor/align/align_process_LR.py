# align_0524/align_process_LR.py
import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import map_coordinates
import pickle

ROOT = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250529_2"
pattern_size = (6, 8)     # Checkerboard pattern size
square_size = 35          # Square size in mm (3.5cm)

def data_parser(index):
    lidar_prefix = "lidar_hybo"
    rgb_prefix = "rgb_arducam"
    lidar_path = os.path.join(ROOT, lidar_prefix)
    rgb_path = os.path.join(ROOT, rgb_prefix)

    return {
        "raw_intensity_path": os.path.join(lidar_path, f"{index:03d}_raw_intensity.png"),
        "raw_depth_path": os.path.join(lidar_path, f"{index:03d}_raw_depth.npy"),
        "lidar_corners": os.path.join(lidar_path, f"corners_{index:03d}.npy"),
        "rgb_img_path": os.path.join(rgb_path, f"{index:03d}.png"),
        "rgb_corners": os.path.join(rgb_path, f"corners_{index:03d}.npy")
    }

def fill_holes_nn(img):
    """Fill zero-valued pixels with nearest valid pixel values."""
    mask = (img == 0)
    if not mask.any():
        return img
    dist, idx = ndimage.distance_transform_edt(mask, return_indices=True)
    filled = img.copy()
    filled[mask] = img[idx[0][mask], idx[1][mask]]
    return filled

def projection_stats(u_rgb, v_rgb, W_r=480, H_r=640):
    inside = (u_rgb >= 0) & (u_rgb < W_r) & (v_rgb >= 0) & (v_rgb < H_r)
    n_all, n_in = len(u_rgb), np.count_nonzero(inside)
    print(f"\u2192 Total {n_all:,d} pts  |  in-bounds {n_in:,d}  ({n_in/n_all:.1%})")
    print(f"   u_rgb min/max: {u_rgb.min():.1f} – {u_rgb.max():.1f}")
    print(f"   v_rgb min/max: {v_rgb.min():.1f} – {v_rgb.max():.1f}")

def project_lidar_to_rgb(depth_lid, inten_lid, R, T, mtx_lid, mtx_rgb,
                         rgb_bgr_img=None, save_path=None):
    H_l, W_l = depth_lid.shape
    u, v = np.meshgrid(np.arange(W_l), np.arange(H_l))
    z = depth_lid.flatten()
    inten = inten_lid.flatten()
    valid = z > 0
    u, v, z, inten = u.flatten()[valid], v.flatten()[valid], z[valid], inten[valid]

    x = (u - mtx_lid[0, 2]) * z / mtx_lid[0, 0]
    y = (v - mtx_lid[1, 2]) * z / mtx_lid[1, 1]
    pts_lid = np.stack([x, y, z], axis=1)

    pts_rgb = (R @ pts_lid.T + T).T
    x_rgb = pts_rgb[:, 0] / pts_rgb[:, 2]
    y_rgb = pts_rgb[:, 1] / pts_rgb[:, 2]
    u_rgb = (x_rgb * mtx_rgb[0, 0] + mtx_rgb[0, 2]).astype(np.int32)
    v_rgb = (y_rgb * mtx_rgb[1, 1] + mtx_rgb[1, 2]).astype(np.int32)

    projection_stats(u_rgb, v_rgb)

    H_r, W_r = 640, 480  # RGB resolution (H, W)
    aligned_d = np.zeros((H_r, W_r), np.float32)
    aligned_i = np.zeros((H_r, W_r), np.uint16)

    for px, py, zval, ival in zip(u_rgb, v_rgb, pts_rgb[:, 2], inten):
        if 0 <= px < W_r and 0 <= py < H_r:
            if aligned_d[py, px] == 0 or aligned_d[py, px] > zval:
                aligned_d[py, px] = zval
                aligned_i[py, px] = ival

    if rgb_bgr_img is not None and save_path is not None:
        overlay = rgb_bgr_img.copy()
        for x_pt, y_pt in zip(u_rgb, v_rgb):
            if 0 <= x_pt < overlay.shape[1] and 0 <= y_pt < overlay.shape[0]:
                cv2.circle(overlay, (int(x_pt), int(y_pt)), 1, (0, 0, 255), -1)
        cv2.imwrite(save_path, overlay)
        print("scatter saved ->", save_path)

    return aligned_d, aligned_i

def main():
    with open("lidar2rgb.pkl", "rb") as f:
        calib = pickle.load(f)
    lid_dict = calib["lid_dict"]
    rgb_dict = calib["rgb_dict"]
    R = calib["R"]
    T = calib["T"]

    idx = 9
    depth = np.load(data_parser(idx)["raw_depth_path"]).astype(np.float32) / 1000.0
    inten = cv2.imread(data_parser(idx)["raw_intensity_path"], cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(data_parser(idx)["rgb_img_path"])

    aligned_d, aligned_i = project_lidar_to_rgb(
        depth, inten, R, T, lid_dict['mtx'], rgb_dict['mtx'],
        rgb_bgr_img=rgb,
        save_path='lidar_scatter.png')

    aligned_d = fill_holes_nn(aligned_d)
    aligned_i = fill_holes_nn(aligned_i)
    vis_i = cv2.normalize(aligned_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_i = cv2.applyColorMap(vis_i, cv2.COLORMAP_MAGMA)
    vis_d = cv2.normalize(aligned_d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_d = cv2.applyColorMap(vis_d, cv2.COLORMAP_JET)
    weighted_i = cv2.addWeighted(vis_i, 0.5, rgb, 0.5, 0)
    weighted_d = cv2.addWeighted(vis_d, 0.5, rgb, 0.5, 0)
    cv2.imwrite(f'LR_aligned_{str(idx)}.png', cv2.hconcat([rgb, weighted_i, weighted_d, vis_i]))

if __name__ == "__main__":
    main()
