import cv2  # OpenCV for image processing
import os  # file path operations
import numpy as np  # numerical computations
import glob  # file pattern matching

# Base directory for sensor data
ROOT = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250525_1"
pattern_size = (6, 8)  # checkerboard cols x rows
square_size = 0.035  # checkerboard square size in meters


def data_parser(idx):
    """Return file paths for lidar and RGB data by index."""  # inline doc
    lidar_dir = os.path.join(ROOT, "lidar_hybo")  # lidar data folder
    rgb_dir = os.path.join(ROOT, "rgb_arducam")  # RGB data folder
    return {
        "raw_intensity_path": os.path.join(lidar_dir, f"{idx:03d}_raw_intensity.png"),  # intensity map
        "raw_depth_path":     os.path.join(lidar_dir, f"{idx:03d}_raw_depth.npy"),      # depth map
        "lidar_corners":      os.path.join(lidar_dir, f"corners_{idx:03d}.npy"),        # lidar checker corners
        "rgb_img_path":       os.path.join(rgb_dir, f"{idx:03d}.png"),                 # RGB image
        "rgb_corners":        os.path.join(rgb_dir, f"corners_{idx:03d}.npy"),          # RGB checker corners
    }


def get_intrinsics():
    """Calibrate and return intrinsics for lidar and RGB cameras."""  # compute camera matrices
    corners = sorted(glob.glob(os.path.join(ROOT, "lidar_hybo", "corners_*.npy")))
    # prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    obj_pts_l, img_pts_l = [], []  # lidar
    obj_pts_r, img_pts_r = [], []  # RGB
    for c in corners:
        idx = int(os.path.basename(c).split('_')[-1].split('.')[0])
        dp = data_parser(idx)
        try:
            lc = np.load(dp['lidar_corners']).reshape(-1, 2)
            rc = np.load(dp['rgb_corners']).reshape(-1, 2)
        except Exception:
            continue
        if lc.shape[0] != objp.shape[0] or rc.shape[0] != objp.shape[0]:
            continue  # skip mismatched
        obj_pts_l.append(objp); img_pts_l.append(lc)
        obj_pts_r.append(objp); img_pts_r.append(rc)
    # sample image for size
    sample_idx = int(os.path.basename(corners[0]).split('_')[-1].split('.')[0])
    dp = data_parser(sample_idx)
    lid_img = cv2.imread(dp['raw_intensity_path'], cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.imread(dp['rgb_img_path'])
    sz_l = (lid_img.shape[1], lid_img.shape[0])  # width, height
    sz_r = (rgb_img.shape[1], rgb_img.shape[0])
    _, m_lid, d_lid, _, _ = cv2.calibrateCamera(obj_pts_l, img_pts_l, sz_l, None, None)
    _, m_rgb, d_rgb, _, _ = cv2.calibrateCamera(obj_pts_r, img_pts_r, sz_r, None, None)
    print("Calibration completed.")
    return {'mtx': m_lid, 'dist': d_lid}, {'mtx': m_rgb, 'dist': d_rgb}


def correlated_points_trust_manual(lid_dict, rgb_dict):
    """Estimate extrinsics R, T using manually annotated corners with minimal filtering."""  # use solvePnPRansac
    m_rgb = np.array(rgb_dict['mtx'], dtype=np.float32)
    dist_rgb = np.array(rgb_dict['dist'], dtype=np.float32)
    m_lid = np.array(lid_dict['mtx'], dtype=np.float32)
    dist_lid = np.array(lid_dict['dist'], dtype=np.float32)
    obj_pts, img_pts = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, 'lidar_hybo', 'corners_*.npy'))):
        idx = int(os.path.basename(f).split('_')[-1].split('.')[0])
        dp = data_parser(idx)
        try:
            lc = np.load(dp['lidar_corners']).reshape(-1, 2)
            rc = np.load(dp['rgb_corners']).reshape(-1, 2)
            depth = np.load(dp['raw_depth_path']).astype(np.float32) / 1000.0
        except Exception:
            continue
        if lc.shape[0] != rc.shape[0]:
            continue  # skip mismatched corners
        for (x_l, y_l), (x_r, y_r) in zip(lc, rc):
            h, w = depth.shape
            if not (0 <= x_l < w and 0 <= y_l < h):
                continue
            z = depth[int(round(y_l)), int(round(x_l))]
            if z <= 0:
                continue
            # back-project to 3D
            X = (x_l - m_lid[0, 2]) * z / m_lid[0, 0]
            Y = (y_l - m_lid[1, 2]) * z / m_lid[1, 1]
            obj_pts.append([X, Y, z]); img_pts.append([x_r, y_r])
    obj = np.float32(obj_pts); img = np.float32(img_pts)
    if len(obj) < 6:
        raise RuntimeError("Insufficient points for PnP.")
    _, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, m_rgb, dist_rgb,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec); T = tvec  # convert rotation
    ratio = len(inliers) / len(obj) if inliers is not None else 0
    print(f"Extrinsics estimated: inlier ratio {ratio:.2%}")
    return R, T


def compute_homography_from_corners(lid_dict, rgb_dict, R, T):
    """Estimate homography via checkerboard corners and extrinsics."""
    m_lid = np.array(lid_dict['mtx'], dtype=np.float32)
    m_rgb = np.array(rgb_dict['mtx'], dtype=np.float32)
    src, dst = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, 'lidar_hybo', 'corners_*.npy'))):
        idx = int(os.path.basename(f).split('_')[-1].split('.')[0])
        dp = data_parser(idx)
        lc = np.load(dp['lidar_corners']).reshape(-1, 2)
        rc = np.load(dp['rgb_corners']).reshape(-1, 2)
        depth = np.load(dp['raw_depth_path']).astype(np.float32) / 1000.0
        for (x_l, y_l), (x_r, y_r) in zip(lc, rc):
            h, w = depth.shape
            if not (0 <= x_l < w and 0 <= y_l < h): continue
            z = depth[int(round(y_l)), int(round(x_l))]
            if z <= 0: continue
            # project lidar to rgb
            X = (x_l - m_lid[0, 2]) * z / m_lid[0, 0]
            Y = (y_l - m_lid[1, 2]) * z / m_lid[1, 1]
            pt = R @ np.array([X, Y, z]) + T.flatten()
            if pt[2] <= 0: continue
            u = pt[0]/pt[2]*m_rgb[0,0] + m_rgb[0,2]
            v = pt[1]/pt[2]*m_rgb[1,1] + m_rgb[1,2]
            if np.hypot(u-x_r, v-y_r) > 50: continue
            src.append([x_l, y_l]); dst.append([x_r, y_r])
    src, dst = np.float32(src), np.float32(dst)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    inl = mask.sum() if mask is not None else 0
    print(f"Homography via corners: inliers {inl}/{len(src)}")
    return H


def visualize_result(src, tgt, H, out_path=None):
    """Warp source to target frame, overlay and save result if requested."""
    warped = cv2.warpPerspective(src, H, (tgt.shape[1], tgt.shape[0]))
    overlay = cv2.addWeighted(tgt, 0.5, warped, 0.5, 0)
    result = cv2.hconcat([tgt, warped, overlay])
    if out_path:
        cv2.imwrite(out_path, result)
        print(f"Saved result: {out_path}")
    return result


def main():
    # 1. Intrinsics
    lid_params, rgb_params = get_intrinsics()
    # 2. Extrinsics via manual corners
    R, T = correlated_points_trust_manual(lid_params, rgb_params)
    # 3. Homography
    H = compute_homography_from_corners(lid_params, rgb_params, R, T)
    print("Homography computed.")
    # 4. Visualize on test frame
    idx = 9
    dp = data_parser(idx)
    li = cv2.imread(dp['raw_intensity_path'], cv2.IMREAD_UNCHANGED)
    rg = cv2.imread(dp['rgb_img_path'])
    visualize_result(li, rg, H, f"homography_{idx:03d}.png")
    print("Processing done.")


if __name__ == '__main__':
    main()
