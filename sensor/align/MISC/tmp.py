import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

def load_calibration_data():
    """저장된 캘리브레이션 데이터 로드"""
    
    # LiDAR to RGB 데이터 로드
    with open('lidar2rgb.pkl', 'rb') as f:
        lidar2rgb_data = pickle.load(f)
    
    # LiDAR to Thermal 데이터 로드  
    with open('lidar2thermal.pkl', 'rb') as f:
        lidar2thermal_data = pickle.load(f)
    
    # Thermal to RGB 데이터 로드
    with open('thermal2rgb.pkl', 'rb') as f:
        thermal2rgb_data = pickle.load(f)
    
    return lidar2rgb_data, lidar2thermal_data, thermal2rgb_data

def compute_homography_from_rt(K1, K2, R, T, plane_normal=None, plane_distance=None):
    """
    R, T로부터 Homography 계산
    
    H = K2 * (R + T*n^T/d) * K1^(-1)
    
    Args:
        K1: Source camera intrinsic matrix
        K2: Target camera intrinsic matrix  
        R: Rotation matrix (3x3)
        T: Translation vector (3x1)
        plane_normal: 평면 법선 벡터 (기본값: [0,0,1])
        plane_distance: 평면까지의 거리 (기본값: 자동 계산)
    
    Returns:
        H: Homography matrix (3x3)
    """
    
    if plane_normal is None:
        plane_normal = np.array([0, 0, 1], dtype=np.float32)  # Z=0 평면 가정
    
    if plane_distance is None:
        # T의 Z 성분을 평균 거리로 사용
        plane_distance = abs(T[2, 0]) if T[2, 0] != 0 else 1.0
    
    # plane_normal을 단위벡터로 정규화
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Homography 계산: H = K2 * (R + T*n^T/d) * K1^(-1)
    T_reshaped = T.reshape(3, 1)
    n_reshaped = plane_normal.reshape(3, 1)
    
    # R + T*n^T/d
    homography_3d = R + (T_reshaped @ n_reshaped.T) / plane_distance
    
    # K2 * homography_3d * K1^(-1)
    K1_inv = np.linalg.inv(K1)
    H = K2 @ homography_3d @ K1_inv
    
    # 정규화 (H[2,2] = 1)
    H = H / H[2, 2]
    
    return H

def compute_multiple_depth_homographies(K1, K2, R, T, depth_range=(0.5, 5.0), num_depths=10):
    """
    여러 깊이에서 homography 계산하여 최적값 찾기
    
    Args:
        K1, K2: Camera intrinsic matrices
        R, T: Rotation and translation
        depth_range: (min_depth, max_depth) 깊이 범위
        num_depths: 테스트할 깊이 개수
        
    Returns:
        homographies: 각 깊이별 homography 리스트
        depths: 사용된 깊이 값들
    """
    
    depths = np.linspace(depth_range[0], depth_range[1], num_depths)
    homographies = []
    
    for depth in depths:
        H = compute_homography_from_rt(K1, K2, R, T, plane_distance=depth)
        homographies.append(H)
    
    return homographies, depths

def analyze_calibration_data():
    """캘리브레이션 데이터 분석"""
    
    lidar2rgb_data, lidar2thermal_data, thermal2rgb_data = load_calibration_data()
    
    print("=== 캘리브레이션 데이터 분석 ===")
    
    # LiDAR to RGB
    print("\n1. LiDAR to RGB:")
    print(f"   Reprojection Error: {lidar2rgb_data['lidar_dict']['reprojection_error']:.4f}")
    print(f"   LiDAR Intrinsic:\n{lidar2rgb_data['lidar_dict']['mtx']}")
    print(f"   RGB Intrinsic:\n{lidar2rgb_data['rgb_dict']['mtx']}")
    print(f"   Translation: {lidar2rgb_data['lidar2rgb_T'].flatten()}")
    
    # LiDAR to Thermal  
    print("\n2. LiDAR to Thermal:")
    print(f"   Reprojection Error: {lidar2thermal_data['lidar_dict']['reprojection_error']:.4f}")
    print(f"   Thermal Intrinsic:\n{lidar2thermal_data['thermal_dict']['mtx']}")
    print(f"   Translation: {lidar2thermal_data['lidar2thermal_T'].flatten()}")
    
    # Thermal to RGB
    print("\n3. Thermal to RGB:")
    print(f"   Translation: {thermal2rgb_data['thermal2rgb_T'].flatten()}")
    
    return lidar2rgb_data, lidar2thermal_data, thermal2rgb_data

def compute_all_homographies():
    """모든 센서 쌍에 대한 homography 계산"""
    
    lidar2rgb_data, lidar2thermal_data, thermal2rgb_data = load_calibration_data()
    
    # Intrinsic matrices
    K_lidar = lidar2rgb_data['lidar_dict']['mtx']
    K_rgb = lidar2rgb_data['rgb_dict']['mtx'] 
    K_thermal = lidar2thermal_data['thermal_dict']['mtx']
    
    # Extrinsic parameters
    R_lidar2rgb = lidar2rgb_data['lidar2rgb_R']
    T_lidar2rgb = lidar2rgb_data['lidar2rgb_T']
    
    R_lidar2thermal = lidar2thermal_data['lidar2thermal_R'] 
    T_lidar2thermal = lidar2thermal_data['lidar2thermal_T']
    
    R_thermal2rgb = thermal2rgb_data['thermal2rgb_R']
    T_thermal2rgb = thermal2rgb_data['thermal2rgb_T']
    
    print("=== Homography 계산 ===")
    
    # 1. LiDAR to RGB Homography
    print("\n1. LiDAR to RGB Homography:")
    avg_depth_lr = abs(T_lidar2rgb[2, 0])
    H_lidar2rgb = compute_homography_from_rt(K_lidar, K_rgb, R_lidar2rgb, T_lidar2rgb, 
                                           plane_distance=avg_depth_lr)
    print(f"   사용된 평면 거리: {avg_depth_lr:.3f}m")
    print(f"   Homography:\n{H_lidar2rgb}")
    
    # 2. LiDAR to Thermal Homography  
    print("\n2. LiDAR to Thermal Homography:")
    avg_depth_lt = abs(T_lidar2thermal[2, 0])
    H_lidar2thermal = compute_homography_from_rt(K_lidar, K_thermal, R_lidar2thermal, T_lidar2thermal,
                                               plane_distance=avg_depth_lt)
    print(f"   사용된 평면 거리: {avg_depth_lt:.3f}m")
    print(f"   Homography:\n{H_lidar2thermal}")
    
    # 3. Thermal to RGB Homography (가장 중요!)
    print("\n3. Thermal to RGB Homography:")
    avg_depth_tr = abs(T_thermal2rgb[2, 0])
    H_thermal2rgb = compute_homography_from_rt(K_thermal, K_rgb, R_thermal2rgb, T_thermal2rgb,
                                             plane_distance=avg_depth_tr)
    print(f"   사용된 평면 거리: {avg_depth_tr:.3f}m")
    print(f"   Homography:\n{H_thermal2rgb}")
    
    # 4. 여러 깊이에서 Thermal to RGB Homography 계산
    print("\n4. 다양한 깊이에서 Thermal to RGB Homography:")
    homographies_tr, depths = compute_multiple_depth_homographies(
        K_thermal, K_rgb, R_thermal2rgb, T_thermal2rgb, 
        depth_range=(0.5, 3.0), num_depths=6)
    
    for i, (H, depth) in enumerate(zip(homographies_tr, depths)):
        print(f"   깊이 {depth:.1f}m에서의 H[0,2], H[1,2]: ({H[0,2]:.1f}, {H[1,2]:.1f})")
    
    return {
        'H_lidar2rgb': H_lidar2rgb,
        'H_lidar2thermal': H_lidar2thermal, 
        'H_thermal2rgb': H_thermal2rgb,
        'H_thermal2rgb_multidepth': homographies_tr,
        'depths': depths,
        'intrinsics': {
            'K_lidar': K_lidar,
            'K_rgb': K_rgb,
            'K_thermal': K_thermal
        }
    }

def apply_homography_transform(image, H, output_shape=None):
    """
    Homography를 사용하여 이미지 변환
    
    Args:
        image: 입력 이미지
        H: Homography matrix (3x3)
        output_shape: 출력 이미지 크기 (width, height)
        
    Returns:
        transformed_image: 변환된 이미지
    """
    
    if output_shape is None:
        output_shape = (image.shape[1], image.shape[0])  # (width, height)
    
    # OpenCV의 warpPerspective 사용
    transformed = cv2.warpPerspective(image, H, output_shape, 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    
    return transformed

def save_homographies(homography_results, filename='homographies.pkl'):
    """계산된 homography들을 파일로 저장"""
    
    with open(filename, 'wb') as f:
        pickle.dump(homography_results, f)
    
    print(f"\nHomography 결과가 {filename}에 저장되었습니다.")

def visualize_transformation_effects(homography_results):
    """Homography 변환 효과 시각화"""
    
    print("\n=== 변환 효과 분석 ===")
    
    # Thermal to RGB homography들 분석
    H_thermal2rgb = homography_results['H_thermal2rgb']
    H_multidepth = homography_results['H_thermal2rgb_multidepth']
    depths = homography_results['depths']
    
    print(f"\n주요 Thermal to RGB Homography (평균 깊이):")
    print(f"  Translation: ({H_thermal2rgb[0,2]:.1f}, {H_thermal2rgb[1,2]:.1f})")
    print(f"  Scale: ({H_thermal2rgb[0,0]:.3f}, {H_thermal2rgb[1,1]:.3f})")
    
    # 깊이별 변환 차이 분석
    print(f"\n깊이별 변환 차이:")
    for i, (H, depth) in enumerate(zip(H_multidepth, depths)):
        tx, ty = H[0,2], H[1,2]
        sx, sy = H[0,0], H[1,1]
        print(f"  {depth:.1f}m: translate=({tx:+6.1f}, {ty:+6.1f}), scale=({sx:.3f}, {sy:.3f})")

def main():
    """메인 실행 함수"""
    
    try:
        # 1. 캘리브레이션 데이터 분석
        analyze_calibration_data()
        
        # 2. 모든 homography 계산
        homography_results = compute_all_homographies()
        
        # 3. 변환 효과 시각화
        visualize_transformation_effects(homography_results)
        
        # 4. 결과 저장
        save_homographies(homography_results)
        
        print("\n=== 사용법 예시 ===")
        print("# Thermal 이미지를 RGB로 변환:")
        print("thermal_img = cv2.imread('thermal.png')")
        print("H = homography_results['H_thermal2rgb']")
        print("aligned_thermal = apply_homography_transform(thermal_img, H, (480, 640))")
        
        print("\n# 특정 깊이의 homography 사용:")
        print("H_depth = homography_results['H_thermal2rgb_multidepth'][2]  # 3번째 깊이")
        print("aligned_thermal = apply_homography_transform(thermal_img, H_depth, (480, 640))")
        
        return homography_results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# =============== 최종 계산된 Homography 매트릭스들 ===============

# 실제 캘리브레이션 데이터로부터 계산된 Thermal to RGB Homography
H_THERMAL2RGB_1M = np.array([
    [1.35459634, 0.08849570, -318.59624500],
    [0.00345546, 1.39081320, 11.76803585],
    [0.00014910, 0.00022732, 1.00000000]
])

H_THERMAL2RGB_2M = np.array([
    [1.23476631, 0.08066721, -237.90381294],
    [0.00314978, 1.26777936, 24.41743490],
    [0.00013591, 0.00020721, 1.00000000]
])

H_THERMAL2RGB_3M = np.array([
    [1.19939930, 0.07835669, -214.08799804],
    [0.00305956, 1.23146678, 28.15081781],
    [0.00013202, 0.00020127, 1.00000000]
])

def apply_thermal_to_rgb_alignment(thermal_img, distance_mode='medium'):
    """
    Thermal 이미지를 RGB 좌표계로 정렬
    
    Args:
        thermal_img: Thermal 이미지 (numpy array)
        distance_mode: 'near' (1m), 'medium' (2m), 'far' (3m)
        
    Returns:
        aligned_thermal: RGB 좌표계로 정렬된 thermal 이미지
    """
    
    # 거리에 따른 Homography 선택
    if distance_mode == 'near':
        H = H_THERMAL2RGB_1M.copy()
    elif distance_mode == 'medium': 
        H = H_THERMAL2RGB_2M.copy()
    elif distance_mode == 'far':
        H = H_THERMAL2RGB_3M.copy()
    else:
        raise ValueError("distance_mode must be 'near', 'medium', or 'far'")
    
    # RGB 이미지 크기 (width=480, height=640)
    rgb_size = (480, 640)
    
    # Homography 적용
    aligned_thermal = cv2.warpPerspective(
        thermal_img, H, rgb_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return aligned_thermal

def create_thermal_rgb_overlay(thermal_img, rgb_img, distance_mode='medium', alpha=0.5):
    """
    Thermal과 RGB 이미지 오버레이
    
    Args:
        thermal_img: Thermal 이미지
        rgb_img: RGB 이미지  
        distance_mode: 거리 모드
        alpha: Thermal 이미지의 투명도 (0~1)
        
    Returns:
        overlay: 오버레이된 이미지
    """
    
    # Thermal 이미지를 RGB 좌표계로 정렬
    aligned_thermal = apply_thermal_to_rgb_alignment(thermal_img, distance_mode)
    
    # Thermal 이미지를 컬러맵 적용 (옵션)
    if len(aligned_thermal.shape) == 2:
        # 16-bit thermal을 8-bit로 정규화
        thermal_normalized = cv2.normalize(aligned_thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 컬러맵 적용 (COLORMAP_INFERNO, COLORMAP_JET 등)
        aligned_thermal_colored = cv2.applyColorMap(thermal_normalized, cv2.COLORMAP_INFERNO)
    else:
        aligned_thermal_colored = aligned_thermal
    
    # RGB 이미지 크기에 맞춤
    if rgb_img.shape[:2] != (640, 480):
        rgb_resized = cv2.resize(rgb_img, (480, 640))
    else:
        rgb_resized = rgb_img
    
    # 오버레이
    overlay = cv2.addWeighted(aligned_thermal_colored, alpha, rgb_resized, 1-alpha, 0)
    
    return overlay, aligned_thermal

def batch_process_thermal_images(thermal_dir, output_dir, distance_mode='medium'):
    """
    여러 thermal 이미지를 일괄 처리
    
    Args:
        thermal_dir: thermal 이미지들이 있는 디렉토리
        output_dir: 정렬된 이미지들을 저장할 디렉토리
        distance_mode: 거리 모드
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    thermal_files = glob.glob(os.path.join(thermal_dir, "*.png")) + \
                   glob.glob(os.path.join(thermal_dir, "*.jpg"))
    
    print(f"Processing {len(thermal_files)} thermal images...")
    
    for thermal_path in thermal_files:
        # Thermal 이미지 로드
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        
        # RGB 좌표계로 정렬
        aligned_thermal = apply_thermal_to_rgb_alignment(thermal_img, distance_mode)
        
        # 저장
        filename = os.path.basename(thermal_path)
        output_path = os.path.join(output_dir, f"aligned_{filename}")
        cv2.imwrite(output_path, aligned_thermal)
        
        print(f"Processed: {filename}")

def evaluate_alignment_quality(thermal_corners, rgb_corners, distance_mode='medium'):
    """
    정렬 품질 평가 (코너 포인트가 있는 경우)
    
    Args:
        thermal_corners: Thermal 이미지의 코너 포인트들 [(u,v), ...]
        rgb_corners: 대응되는 RGB 이미지의 코너 포인트들 [(u,v), ...]
        distance_mode: 거리 모드
        
    Returns:
        rmse: Root Mean Square Error
        max_error: 최대 오차
    """
    
    if distance_mode == 'near':
        H = H_THERMAL2RGB_1M
    elif distance_mode == 'medium':
        H = H_THERMAL2RGB_2M  
    elif distance_mode == 'far':
        H = H_THERMAL2RGB_3M
    
    errors = []
    
    for (u_t, v_t), (u_r, v_r) in zip(thermal_corners, rgb_corners):
        # Thermal 포인트를 RGB 좌표계로 변환
        thermal_point = np.array([u_t, v_t, 1])
        transformed_point = H @ thermal_point
        
        # 정규화
        u_pred = transformed_point[0] / transformed_point[2]
        v_pred = transformed_point[1] / transformed_point[2]
        
        # 오차 계산
        error = np.sqrt((u_pred - u_r)**2 + (v_pred - v_r)**2)
        errors.append(error)
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    
    return rmse, max_error

# =============== 사용 예시 ===============

def example_usage():
    """사용 예시"""
    
    print("=== Thermal to RGB Alignment 사용 예시 ===")
    
    # 1. 단일 이미지 정렬
    thermal_img = cv2.imread('/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/ir.png', cv2.IMREAD_UNCHANGED)
    thermal_img = cv2.resize(thermal_img, (640, 480))  # RGB 크기에 맞춤
    aligned_thermal = apply_thermal_to_rgb_alignment(thermal_img, 'far')
    cv2.imwrite('aligned_thermal.png', aligned_thermal)
    
    # 2. RGB와 오버레이
    print("\n2. RGB와 오버레이:")
    print("rgb_img = cv2.imread('rgb.png')")
    print("overlay, aligned_thermal = create_thermal_rgb_overlay(thermal_img, rgb_img, 'medium')")
    print("cv2.imwrite('overlay.png', overlay)")
    
    # 3. 일괄 처리
    print("\n3. 일괄 처리:")
    print("batch_process_thermal_images('thermal_images/', 'aligned_images/', 'medium')")
    
    # 4. 거리별 Homography 직접 사용
    print("\n4. 거리별 Homography 직접 사용:")
    print("# 가까운 거리 (1m): H_THERMAL2RGB_1M")
    print("# 중간 거리 (2m): H_THERMAL2RGB_2M") 
    print("# 먼 거리 (3m): H_THERMAL2RGB_3M")
    print("aligned = cv2.warpPerspective(thermal_img, H_THERMAL2RGB_2M, (480, 640))")

if __name__ == "__main__":
    # 실제 데이터 분석
    # results = main()
    
    # 사용 예시 출력
    example_usage()