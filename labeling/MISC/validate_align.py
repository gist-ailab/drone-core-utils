import numpy as np
import os
import cv2
import glob

# 경로 설정
BASE_DIR = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco_synced/images_heuristic_cropped_400x381'

def get_all_modality_dirs():
    """모든 모달리티 디렉토리 경로를 반환"""
    modality_dirs = {
        'rgb': os.path.join(BASE_DIR, 'group_rgb'),
        'ir': os.path.join(BASE_DIR, 'group_ir'), 
        'depth': os.path.join(BASE_DIR, 'group_depth'),
        'intensity': os.path.join(BASE_DIR, 'group_intensity')
    }
    
    # 디렉토리 존재 확인
    existing_dirs = {}
    for modality, path in modality_dirs.items():
        if os.path.exists(path):
            existing_dirs[modality] = path
            print(f"✓ Found {modality}: {path}")
        else:
            print(f"✗ Missing {modality}: {path}")
    
    return existing_dirs

def get_all_image_files(modality_dirs):
    """각 모달리티의 모든 이미지 파일들을 수집"""
    all_files = {}
    
    for modality, dir_path in modality_dirs.items():
        files = sorted(glob.glob(os.path.join(dir_path, '**', '*.png'), recursive=True))
        all_files[modality] = files
        print(f"{modality}: {len(files)} files found")
    
    return all_files

def find_matching_files_by_filename(rgb_file, all_files):
    """RGB 파일명을 기준으로 다른 모달리티의 매칭 파일들을 찾음"""
    rgb_filename = os.path.basename(rgb_file)
    
    matches = {'rgb': rgb_file}
    
    # 각 모달리티에서 같은 파일명 찾기
    for modality in ['ir', 'depth', 'intensity']:
        if modality not in all_files:
            matches[modality] = None
            continue
            
        found = False
        for file_path in all_files[modality]:
            if os.path.basename(file_path) == rgb_filename:
                matches[modality] = file_path
                found = True
                break
        
        if not found:
            matches[modality] = None
    
    return matches

def create_overlay_visualization(matches):
    """4개 모달리티의 시각화를 생성: RGB, RGB+IR, RGB+Depth, RGB+Intensity"""
    
    # RGB 이미지 로드 (기준)
    rgb_img = cv2.imread(matches['rgb']) if matches['rgb'] else None
    if rgb_img is None:
        print(f"Failed to load RGB image: {matches['rgb']}")
        return None
    
    h, w = rgb_img.shape[:2]
    
    # 각 모달리티 이미지 로드
    ir_img = cv2.imread(matches['ir']) if matches['ir'] else None
    depth_img = cv2.imread(matches['depth']) if matches['depth'] else None
    intensity_img = cv2.imread(matches['intensity']) if matches['intensity'] else None
    
    # 1. RGB 원본
    rgb_display = rgb_img.copy()
    cv2.putText(rgb_display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 2. RGB + IR overlay
    if ir_img is not None:
        # IR 이미지 크기를 RGB와 맞춤
        if ir_img.shape[:2] != (h, w):
            ir_img = cv2.resize(ir_img, (w, h))
        rgb_ir_overlay = cv2.addWeighted(rgb_img, 0.3, ir_img, 0.7, 0)
        cv2.putText(rgb_ir_overlay, "RGB + IR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        rgb_ir_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_ir_overlay, "IR not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 3. RGB + Depth overlay
    if depth_img is not None:
        # Depth 이미지 크기를 RGB와 맞춤
        if depth_img.shape[:2] != (h, w):
            depth_img = cv2.resize(depth_img, (w, h))
        rgb_depth_overlay = cv2.addWeighted(rgb_img, 0.3, depth_img, 0.7, 0)
        cv2.putText(rgb_depth_overlay, "RGB + Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        rgb_depth_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_depth_overlay, "Depth not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 4. RGB + Intensity overlay
    if intensity_img is not None:
        # Intensity 이미지 크기를 RGB와 맞춤
        if intensity_img.shape[:2] != (h, w):
            intensity_img = cv2.resize(intensity_img, (w, h))
        rgb_intensity_overlay = cv2.addWeighted(rgb_img, 0.3, intensity_img, 0.7, 0)
        cv2.putText(rgb_intensity_overlay, "RGB + Intensity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        rgb_intensity_overlay = np.zeros_like(rgb_img)
        cv2.putText(rgb_intensity_overlay, "Intensity not found", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 4개 이미지를 2x2로 배치
    top_row = cv2.hconcat([rgb_display, rgb_ir_overlay])
    bottom_row = cv2.hconcat([rgb_depth_overlay, rgb_intensity_overlay])
    combined = cv2.vconcat([top_row, bottom_row])
    
    return combined

def main():
    """메인 실행 함수"""
    print("="*60)
    print("ALIGNMENT VALIDATION TOOL")
    print("="*60)
    print(f"Data source: {BASE_DIR}")
    print()
    
    # 1. 모달리티 디렉토리 확인
    modality_dirs = get_all_modality_dirs()
    if not modality_dirs:
        print("No modality directories found!")
        return
    
    if 'rgb' not in modality_dirs:
        print("RGB directory not found! Cannot proceed.")
        return
    
    print()
    
    # 2. 모든 이미지 파일 수집
    print("Collecting image files...")
    all_files = get_all_image_files(modality_dirs)
    
    if not all_files.get('rgb'):
        print("No RGB files found!")
        return
    
    rgb_files = all_files['rgb']
    print(f"\nTotal RGB files: {len(rgb_files)}")
    
    # 3. 시각화 시작
    print("\nControls:")
    print("  'd' or 'D' or Right Arrow: Next image")
    print("  'a' or 'A' or Left Arrow: Previous image")
    print("  's' or 'S': Save current visualization")
    print("  'q' or 'Q' or ESC: Quit")
    print("  'i' or 'I': Show image info")
    
    current_idx = 0
    window_name = 'Alignment Validation - RGB | RGB+IR | RGB+Depth | RGB+Intensity'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 800)
    
    while True:
        # 현재 RGB 파일
        current_rgb_file = rgb_files[current_idx]
        
        # 매칭되는 다른 모달리티 파일들 찾기
        matches = find_matching_files_by_filename(current_rgb_file, all_files)
        
        # 시각화 이미지 생성
        vis_img = create_overlay_visualization(matches)
        
        if vis_img is None:
            print(f"Failed to create visualization for {current_rgb_file}")
            current_idx = (current_idx + 1) % len(rgb_files)
            continue
        
        # 현재 이미지 정보 표시
        rgb_filename = os.path.basename(current_rgb_file)
        group_info = current_rgb_file.split(os.sep)[-2]  # group_xx 정보
        
        info_text = f"[{current_idx + 1}/{len(rgb_files)}] {group_info}/{rgb_filename}"
        cv2.putText(vis_img, info_text, (10, vis_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
            save_name = f"validation_{current_idx:04d}_{group_info}_{os.path.splitext(rgb_filename)[0]}.png"
            cv2.imwrite(save_name, vis_img)
            print(f"Saved: {save_name}")
        elif key == ord('i') or key == ord('I'):  # I - Info
            print(f"\n--- Image Info [{current_idx + 1}/{len(rgb_files)}] ---")
            print(f"RGB: {current_rgb_file}")
            for modality in ['ir', 'depth', 'intensity']:
                if matches[modality]:
                    print(f"{modality.upper()}: {matches[modality]}")
                else:
                    print(f"{modality.upper()}: NOT FOUND")
            print("-" * 50)
    
    cv2.destroyAllWindows()
    print("\nValidation completed.")

if __name__ == "__main__":
    main()
