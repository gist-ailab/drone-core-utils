# align_0524/align_process_LT2.py

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage            # Needed for nearest‑neighbour hole fill
from scipy.ndimage import map_coordinates

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ROOT = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250526_1"
pattern_size = (6, 8)               # Chessboard pattern (cols, rows)
square_size = 35                    # Square size in mm (3.5 cm)


# ----------------------------------------------------------------------
# Helper: build all relevant paths for a frame index
# ----------------------------------------------------------------------
def data_parser(index: int):
    lidar_prefix = "lidar_hybo"
    rgb_prefix   = "thermal_pure_resized"          # Thermal RGB images

    lidar_path = os.path.join(ROOT, lidar_prefix)
    rgb_path   = os.path.join(ROOT, rgb_prefix)

    return {
        "raw_intensity_path": os.path.join(lidar_path, f"{index:03d}_raw_intensity.png"),
        "raw_depth_path":     os.path.join(lidar_path, f"{index:03d}_raw_depth.npy"),
        "lidar_corners":      os.path.join(lidar_path, f"corners_{index:03d}.npy"),
        "rgb_img_path":       os.path.join(rgb_path,   f"{index:03d}.png"),
        "rgb_corners":        os.path.join(rgb_path,   f"corners_{index:03d}.npy"),
    }


# ----------------------------------------------------------------------
# Strict corner filtering & PnP with multiple RANSAC attempts
# ----------------------------------------------------------------------
def correlated_points_improved(lid_dict: dict, rgb_dict: dict):
    mtx_rgb  = np.asarray(rgb_dict["mtx"], dtype=np.float32)
    dist_rgb = np.asarray(rgb_dict["dist"], dtype=np.float32)
    mtx_lid  = np.asarray(lid_dict["mtx"], dtype=np.float32)

    obj_pts, img_pts = [], []

    for path in sorted(glob.glob(os.path.join(ROOT, "lidar_hybo", "corners_*.npy"))):
        idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
        p   = data_parser(idx)

        lid_c = np.load(p["lidar_corners"]).reshape(-1, 2)
        rgb_c = np.load(p["rgb_corners"]).reshape(-1, 2)
        depth = np.load(p["raw_depth_path"]).astype(np.float32) / 1000.0  # mm→m
        H, W  = depth.shape

        # 1) Ensure the expected number of corners
        if lid_c.shape[0] != rgb_c.shape[0] or lid_c.shape[0] != pattern_size[0] * pattern_size[1]:
            print(f"[SKIP] {idx:03d} – corner count mismatch")
            continue

        valid_pairs, valid_depths = [], []

        for (u_l, v_l), (u_r, v_r) in zip(lid_c, rgb_c):
            u_i, v_i = int(round(u_l)), int(round(v_l))

            # 2) Guard against borders (margin=2 px)
            if not (2 <= u_i < W - 2 and 2 <= v_i < H - 2):
                continue

            # 3) Robust depth: take median in 3×3 window
            neighbourhood = [
                depth[v_i + dv, u_i + du]
                for dv in (-1, 0, 1)
                for du in (-1, 0, 1)
                if 0 <= u_i + du < W and 0 <= v_i + dv < H and depth[v_i + dv, u_i + du] > 0
            ]

            if len(neighbourhood) < 3:
                continue
            z = float(np.median(neighbourhood))

            # 4) Accept only 0.1 m – 10 m
            if not (0.1 < z <= 10.0):
                continue

            # Approximate 3‑D position in LiDAR frame
            x = (u_l - mtx_lid[0, 2]) * z / mtx_lid[0, 0]
            y = (v_l - mtx_lid[1, 2]) * z / mtx_lid[1, 1]

            # 5) Ensure RGB corner is comfortably inside bounds
            if not (10 <= u_r < 630 and 10 <= v_r < 470):
                continue

            valid_pairs.append(((u_l, v_l), (u_r, v_r), x, y, z))
            valid_depths.append(z)

        # 6) Require ≥70 % valid corners in this image
        if len(valid_pairs) < pattern_size[0] * pattern_size[1] * 0.7:
            print(f"[SKIP] {idx:03d} – insufficient valid corners ({len(valid_pairs)})")
            continue

        # 7) Depth variance check
        depths_arr = np.asarray(valid_depths)
        if np.std(depths_arr) > np.mean(depths_arr) * 0.10:
            print(f"[SKIP] {idx:03d} – depth std too high")
            continue

        print(f"[USE]  {idx:03d} – {len(valid_pairs)} valid pts, mean depth {np.mean(depths_arr):.2f} m")

        for (u_l, v_l), (u_r, v_r), x, y, z in valid_pairs:
            obj_pts.append([x, y, z])
            img_pts.append([u_r, v_r])

    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)
    print(f"[INFO] total correspondences after filtering: {len(obj_pts)}")

    if len(obj_pts) < 20:
        raise RuntimeError("Not enough correspondences for PnP.")

    # --- Multiple PnP RANSAC attempts
    best_ratio, best_R, best_T = 0.0, None, None
    for attempt in range(5):
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts, imagePoints=img_pts,
            cameraMatrix=mtx_rgb, distCoeffs=dist_rgb,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0, iterationsCount=50_000, confidence=0.995
        )
        if not ok or inliers is None:
            continue
        ratio = len(inliers) / len(obj_pts)
        print(f"  attempt {attempt+1}: inlier ratio {ratio:.2%}")
        if ratio > best_ratio:
            best_ratio = ratio
            best_R, _ = cv2.Rodrigues(rvec)
            best_T    = tvec.reshape(3, 1)
            best_inliers = inliers

    if best_R is None:
        raise RuntimeError("PnP RANSAC failed on all attempts.")

    print(f"[INFO] best inlier ratio {best_ratio:.2%} ({len(best_inliers)}/{len(obj_pts)})")

    # --- Optional refinement on inliers
    if len(best_inliers) > 10:
        in_obj = obj_pts[best_inliers.flatten()]
        in_img = img_pts[best_inliers.flatten()]
        ok, rvec_ref, tvec_ref = cv2.solvePnP(in_obj, in_img, mtx_rgb, dist_rgb, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            best_R, _ = cv2.Rodrigues(rvec_ref)
            best_T    = tvec_ref.reshape(3, 1)
            print("[INFO] pose refined using inliers")

    return best_R, best_T



if __name__ == "__main__":
    main()