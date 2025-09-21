import numpy as np
import os
import cv2
import glob

# 경로 설정
SAVE_BASE_DIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco/images'
GROUP_RGB_DIR = os.path.join(SAVE_BASE_DIR, 'group_rgb_wocalib')
GROUP_DEPTH_DIR = os.path.join(SAVE_BASE_DIR, 'group_depth')
GROUP_INTENSITY_DIR = os.path.join(SAVE_BASE_DIR, 'group_intensity')
GROUP_IR_DIR = os.path.join(SAVE_BASE_DIR, 'group_ir')

def get_all_aligned_files():
    """모든 align된 이미지 파일들의 경로를 가져오는 함수"""
    rgb_files = sorted(glob.glob(os.path.join(GROUP_RGB_DIR, '**', '*.png'), recursive=True))
    depth_files = sorted(glob.glob(os.path.join(GROUP_DEPTH_DIR, '**', '*.png'), recursive=True))
    intensity_files = sorted(glob.glob(os.path.join(GROUP_INTENSITY_DIR, '**', '*.png'), recursive=True))
    ir_files = sorted(glob.glob(os.path.join(GROUP_IR_DIR, '**', '*.png'), recursive=True))
    
    return rgb_files, depth_files, intensity_files, ir_files

def find_matching_files(rgb_path, depth_files, intensity_files, ir_files):
    """RGB 파일명을 기준으로 매칭되는 다른 모달리티 파일들을 찾는 함수"""
    rgb_filename = os.path.basename(rgb_path)
    
    # 매칭되는 파일들 찾기
    depth_match = None
    intensity_match = None
    ir_match = None
    
    for depth_file in depth_files:
        if os.path.basename(depth_file) == rgb_filename:
            depth_match = depth_file
            break
    
    for intensity_file in intensity_files:
        if os.path.basename(intensity_file) == rgb_filename:
            intensity_match = intensity_file
            break
    
    for ir_file in ir_files:
        if os.path.basename(ir_file) == rgb_filename:
            ir_match = ir_file
            break
    
    return depth_match, intensity_match, ir_match

def create_visualization(rgb_path, depth_path, intensity_path, ir_path):
    """4개 모달리티의 시각화 이미지를 생성하는 함수"""
    # 이미지 로드
    rgb_img = cv2.imread(rgb_path) if rgb_path else None
    depth_img = cv2.imread(depth_path) if depth_path else None
    intensity_img = cv2.imread(intensity_path) if intensity_path else None
    ir_img = cv2.imread(ir_path) if ir_path else None
    
    if rgb_img is None:
        print(f"Failed to load RGB image: {rgb_path}")
        return None
    
    h, w = rgb_img.shape[:2]
    
    # 1. RGB 원본
    rgb_display = rgb_img.copy()
    
    # 2. RGB + IR overlay
    if ir_img is not None:
        rgb_ir_overlay = cv2.addWeighted(rgb_img, 0.6, ir_img, 0.4, 0)
    else:
        rgb_ir_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_ir_overlay, "IR not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 3. RGB + Depth overlay
    if depth_img is not None:
        rgb_depth_overlay = cv2.addWeighted(rgb_img, 0.6, depth_img, 0.4, 0)
    else:
        rgb_depth_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_depth_overlay, "Depth not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 4. RGB + Intensity overlay
    if intensity_img is not None:
        rgb_intensity_overlay = cv2.addWeighted(rgb_img, 0.6, intensity_img, 0.4, 0)
    else:
        rgb_intensity_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_intensity_overlay, "Intensity not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 텍스트 추가
    cv2.putText(rgb_display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rgb_ir_overlay, "RGB + IR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rgb_depth_overlay, "RGB + Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rgb_intensity_overlay, "RGB + Intensity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 4개 이미지를 2x2로 배치
    top_row = cv2.hconcat([rgb_display, rgb_ir_overlay])
    bottom_row = cv2.hconcat([rgb_depth_overlay, rgb_intensity_overlay])
    combined = cv2.vconcat([top_row, bottom_row])
    
    return combined

def main():
    """메인 실행 함수"""
    print("Loading aligned image files...")
    rgb_files, depth_files, intensity_files, ir_files = get_all_aligned_files()
    
    if not rgb_files:
        print("No RGB files found! Please check the path.")
        return
    
    print(f"Found {len(rgb_files)} RGB files")
    print("Controls:")
    print("  'd' or 'D' or Right Arrow: Next image")
    print("  'a' or 'A' or Left Arrow: Previous image")
    print("  'q' or 'Q' or ESC: Quit")
    print("  's' or 'S': Save current visualization")
    
    current_idx = 0
    window_name = 'Alignment Validation - RGB | RGB+IR | RGB+Depth | RGB+Intensity'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    while True:
        # 현재 RGB 파일
        rgb_path = rgb_files[current_idx]
        
        # 매칭되는 다른 모달리티 파일들 찾기
        depth_path, intensity_path, ir_path = find_matching_files(rgb_path, depth_files, intensity_files, ir_files)
        
        # 시각화 이미지 생성
        vis_img = create_visualization(rgb_path, depth_path, intensity_path, ir_path)
        
        if vis_img is None:
            print(f"Failed to create visualization for {rgb_path}")
            current_idx = (current_idx + 1) % len(rgb_files)
            continue
        
        # 현재 이미지 정보 표시
        info_text = f"[{current_idx + 1}/{len(rgb_files)}] {os.path.basename(rgb_path)}"
        cv2.putText(vis_img, info_text, (10, vis_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, vis_img)
        
        # 키보드 입력 처리
        key = cv2.waitKeyEx(0)
        
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            break
        elif key == ord('d') or key == ord('D') or key == 2555904:  # D or Right Arrow
            current_idx = (current_idx + 1) % len(rgb_files)
        elif key == ord('a') or key == ord('A') or key == 2424832:  # A or Left Arrow
            current_idx = (current_idx - 1) % len(rgb_files)
        elif key == ord('s') or key == ord('S'):  # S - Save
            save_name = f"validation_{current_idx:04d}_{os.path.splitext(os.path.basename(rgb_path))[0]}.png"
            cv2.imwrite(save_name, vis_img)
            print(f"Saved: {save_name}")
    
    cv2.destroyAllWindows()
    print("Validation completed.")

if __name__ == "__main__":
    main()