import cv2
import numpy as np
import glob
import os
from pathlib import Path

def resize_image_and_points(image_path, corners_path, target_height=640, target_width=480, output_dir=None):
    """
    이미지와 체커보드 포인트를 함께 resize하는 함수
    
    Args:
        image_path (str): 이미지 파일 경로
        corners_path (str): 체커보드 포인트 numpy 파일 경로
        target_height (int): 목표 높이 (기본값: 640)
        target_width (int): 목표 너비 (기본값: 480)
        output_dir (str): 출력 디렉토리 (None이면 원본 디렉토리에 '_resized' 추가)
    
    Returns:
        tuple: (resized_image, resized_corners)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 원본 이미지 크기
    original_height, original_width = image.shape[:2]
    print(f"원본 이미지 크기: {original_width} x {original_height}")
    
    # 체커보드 포인트 로드
    corners = np.load(corners_path)
    print(f"체커보드 포인트 shape: {corners.shape}")
    
    # 이미지 resize
    resized_image = cv2.resize(image, (target_width, target_height))
    
    # 스케일링 비율 계산
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    print(f"스케일링 비율: x={scale_x:.4f}, y={scale_y:.4f}")
    
    # 체커보드 포인트 resize
    # (48, 1, 2) 형태에서 (48, 2) 형태로 변환하여 처리
    resized_corners = corners.copy()
    resized_corners[:, :, 0] *= scale_x  # x 좌표
    resized_corners[:, :, 1] *= scale_y  # y 좌표
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(image_path) + "_resized"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 추출
    image_name = os.path.basename(image_path)
    corners_name = os.path.basename(corners_path)
    
    # 파일 저장
    output_image_path = os.path.join(output_dir, image_name)
    output_corners_path = os.path.join(output_dir, corners_name)
    
    cv2.imwrite(output_image_path, resized_image)
    np.save(output_corners_path, resized_corners)
    
    print(f"저장완료:")
    print(f"  이미지: {output_image_path}")
    print(f"  포인트: {output_corners_path}")
    
    return resized_image, resized_corners

def visualize_corners(image, corners, title="Checkerboard Corners", save_path=None):
    """
    이미지에 체커보드 포인트를 시각화하는 함수
    
    Args:
        image: 이미지 배열
        corners: 체커보드 포인트 배열 (48, 1, 2)
        title: 윈도우 제목
        save_path: 저장 경로 (None이면 저장하지 않음)
    """
    vis_image = image.copy()
    
    # 포인트들을 이미지에 그리기
    for i, corner in enumerate(corners):
        x, y = int(corner[0][0]), int(corner[0][1])
        cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis_image, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        print(f"시각화 이미지 저장: {save_path}")
    
    return vis_image

def find_matching_files(base_dir):
    """
    디렉토리에서 이미지와 numpy 파일의 매칭 쌍을 찾는 함수
    
    Args:
        base_dir (str): 기본 디렉토리 경로
    
    Returns:
        list: [(image_path, corners_path), ...] 형태의 매칭된 파일 쌍 리스트
    """
    # 이미지 파일들 찾기
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(base_dir, ext)))
        image_files.extend(glob.glob(os.path.join(base_dir, ext.upper())))
    
    # numpy 파일들 찾기
    numpy_files = glob.glob(os.path.join(base_dir, '*.npy'))
    
    print(f"찾은 이미지 파일: {len(image_files)}개")
    print(f"찾은 numpy 파일: {len(numpy_files)}개")
    
    # 매칭된 파일 쌍 저장
    matched_pairs = []
    
    for image_path in sorted(image_files):
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        
        # 여러 매칭 방식 시도
        corners_path = None
        
        # 방식 1: 숫자로 매칭
        image_nums = ''.join(filter(str.isdigit, image_base))
        for numpy_path in numpy_files:
            numpy_base = os.path.splitext(os.path.basename(numpy_path))[0]
            if 'corners' in numpy_base.lower():
                numpy_nums = ''.join(filter(str.isdigit, numpy_base))
                if image_nums == numpy_nums and image_nums:
                    corners_path = numpy_path
                    break
        
        # 방식 2: 파일명 직접 매칭 (예: image_001.jpg -> corners_001.npy)
        if not corners_path:
            expected_corners_name = image_base.replace('image', 'corners') + '.npy'
            potential_path = os.path.join(base_dir, expected_corners_name)
            if os.path.exists(potential_path):
                corners_path = potential_path
        
        # 방식 3: 유사한 이름 찾기
        if not corners_path:
            for numpy_path in numpy_files:
                numpy_base = os.path.splitext(os.path.basename(numpy_path))[0]
                if image_base.lower() in numpy_base.lower() or numpy_base.lower() in image_base.lower():
                    corners_path = numpy_path
                    break
        
        if corners_path:
            matched_pairs.append((image_path, corners_path))
            print(f"매칭: {os.path.basename(image_path)} ↔ {os.path.basename(corners_path)}")
        else:
            print(f"매칭 실패: {os.path.basename(image_path)}")
    
    return matched_pairs

def batch_resize_thermal_data(base_dir, target_height=640, target_width=480, visualize=False):
    """
    thermal 폴더의 모든 이미지와 체커보드 포인트를 일괄 resize하는 함수
    
    Args:
        base_dir (str): 기본 디렉토리 경로
        target_height (int): 목표 높이
        target_width (int): 목표 너비
        visualize (bool): 시각화 이미지 생성 여부
    """
    # 매칭된 파일 쌍 찾기
    matched_pairs = find_matching_files(base_dir)
    
    if not matched_pairs:
        print("처리할 매칭된 파일 쌍이 없습니다.")
        return
    
    print(f"\n총 {len(matched_pairs)}개의 파일 쌍을 처리합니다.")
    
    processed_count = 0
    output_dir = base_dir + "_resized"
    
    for image_path, corners_path in matched_pairs:
        try:
            print(f"\n처리 중: {os.path.basename(image_path)} + {os.path.basename(corners_path)}")
            
            # resize 수행
            resized_image, resized_corners = resize_image_and_points(
                image_path, corners_path, target_height, target_width, output_dir
            )
            
            # 시각화 이미지 생성 (옵션)
            if visualize:
                vis_dir = os.path.join(output_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)
                
                # 원본 시각화
                original_image = cv2.imread(image_path)
                original_corners = np.load(corners_path)
                original_vis_path = os.path.join(vis_dir, f"original_{os.path.basename(image_path)}")
                visualize_corners(original_image, original_corners, save_path=original_vis_path)
                
                # resize된 이미지 시각화
                resized_vis_path = os.path.join(vis_dir, f"resized_{os.path.basename(image_path)}")
                visualize_corners(resized_image, resized_corners, save_path=resized_vis_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"오류 발생: {e}")
    
    print(f"\n총 {processed_count}개 파일 쌍이 성공적으로 처리되었습니다.")
    print(f"출력 디렉토리: {output_dir}")

# 사용 예시
if __name__ == "__main__":
    # 단일 파일 처리 예시
    # resize_image_and_points("image_001.jpg", "corners_001.npy")
    
    # 배치 처리 예시 (시각화 포함)
    thermal_dir = "/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intrinsic_images_250601_undistorted/thermal_pure"
    batch_resize_thermal_data(thermal_dir, target_height=480, target_width=640, visualize=True)
    
    # 빠른 배치 처리 (시각화 없이)
    # batch_resize_thermal_data(thermal_dir, target_height=640, target_width=480, visualize=False)