import os
import json
import cv2
from tqdm import tqdm
import re

def read_crop_coordinates(coord_file_path):
    """
    crop_coordinates.txt 파일에서 좌표를 읽어오는 함수
    """
    if not os.path.exists(coord_file_path):
        print(f"Coordinates file not found: {coord_file_path}")
        return None
    
    try:
        with open(coord_file_path, 'r') as f:
            content = f.read()
        
        # x1, y1 = 37, 107 형태의 패턴 찾기
        x1_y1_match = re.search(r'x1,\s*y1\s*=\s*(\d+),\s*(\d+)', content)
        x2_y2_match = re.search(r'x2,\s*y2\s*=\s*(\d+),\s*(\d+)', content)
        
        if x1_y1_match and x2_y2_match:
            x1, y1 = int(x1_y1_match.group(1)), int(x1_y1_match.group(2))
            x2, y2 = int(x2_y2_match.group(1)), int(x2_y2_match.group(2))
            
            print(f"Loaded crop coordinates: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"Crop size: {x2-x1}x{y2-y1}")
            return x1, y1, x2, y2
        else:
            print("Could not parse coordinates from file")
            return None
            
    except Exception as e:
        print(f"Error reading coordinates file: {e}")
        return None

def get_user_input():
    """
    사용자로부터 설정 정보를 입력받는 함수
    """
    print("="*60)
    print("INTERACTIVE CROP DATASET TOOL")
    print("="*60)
    
    # 1. 좌표 파일 경로
    default_coord_path = "/media/ailab/HDD1/Workspace/src/Project/Drone24/drone-core-utils/labeling/MISC/crop_coordinates.txt"
    coord_path = input(f"Enter crop coordinates file path\n(default: {default_coord_path})\n> ").strip()
    if not coord_path:
        coord_path = default_coord_path
    
    # 2. 좌표 읽기
    coords = read_crop_coordinates(coord_path)
    if coords is None:
        print("Failed to load coordinates. Exiting.")
        return None
    
    x1, y1, x2, y2 = coords
    
    # 3. ROOT 디렉토리
    default_root = "/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco_synced"
    root_dir = input(f"Enter dataset ROOT directory\n(default: {default_root})\n> ").strip()
    if not root_dir:
        root_dir = default_root
    
    # 4. 입력 폴더 이름
    default_input = "images_heuristic"
    input_folder = input(f"Enter input folder name\n(default: {default_input})\n> ").strip()
    if not input_folder:
        input_folder = default_input
    
    # 5. 출력 폴더 이름
    default_output = f"images_heuristic_cropped_{x2-x1}x{y2-y1}"
    output_folder = input(f"Enter output folder name\n(default: {default_output})\n> ").strip()
    if not output_folder:
        output_folder = default_output
    
    # 6. JSON 파일 이름
    default_json = "train_filtered.json"
    json_name = input(f"Enter input JSON filename\n(default: {default_json})\n> ").strip()
    if not json_name:
        json_name = default_json
    
    # 7. 출력 JSON 파일 이름
    json_base = os.path.splitext(json_name)[0]
    default_output_json = f"{json_base}_cropped_{x2-x1}x{y2-y1}.json"
    output_json = input(f"Enter output JSON filename\n(default: {default_output_json})\n> ").strip()
    if not output_json:
        output_json = default_output_json
    
    config = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'root_dir': root_dir,
        'input_folder': input_folder,
        'output_folder': output_folder,
        'json_path': os.path.join(root_dir, 'labels', json_name),
        'output_json_path': os.path.join(root_dir, 'labels', output_json)
    }
    
    # 설정 확인
    print("\n" + "-"*60)
    print("CONFIGURATION SUMMARY:")
    print("-"*60)
    print(f"Crop coordinates: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"Crop size: {x2-x1}x{y2-y1}")
    print(f"ROOT directory: {root_dir}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Input JSON: {config['json_path']}")
    print(f"Output JSON: {config['output_json_path']}")
    print("-"*60)
    
    confirm = input("Proceed with these settings? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return None
    
    return config


def crop_and_filter_coco(config):
    """
    COCO 데이터셋의 이미지를 자르고, 그에 맞게 Annotation을 필터링 및 수정합니다.
    잘린 영역에 걸치는 바운딩 박스는 좌표를 재계산합니다.
    """
    # 설정에서 변수 추출
    x1, y1, x2, y2 = config['x1'], config['y1'], config['x2'], config['y2']
    ROOT = config['root_dir']
    subfolder_prefix = config['input_folder']
    save_folder_prefix = config['output_folder']
    JSON_PATH = config['json_path']
    SAVE_JSON_PATH = config['output_json_path']
    
    # 1. JSON 파일 로드
    print(f"'{JSON_PATH}' 파일을 로드합니다...")
    with open(JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    # 새로운 COCO 데이터를 담을 딕셔너리 준비
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': [],
        'annotations': []
    }

    new_width = x2 - x1
    new_height = y2 - y1

    print(f"\n총 {len(coco_data['images'])}개의 이미지를 처리합니다...")
    
    successfully_processed_image_ids = set()

    for img_info in tqdm(coco_data['images'], desc="Cropping images"):
        paths_to_crop = {
            'file_name': img_info.get('file_name'),
            'depth_path': img_info.get('depth_path'),
            'lidar_path': img_info.get('lidar_path'),
            'event_path': img_info.get('event_path')
        }
        
        all_files_saved_successfully = True

        for key, relative_path in paths_to_crop.items():
            if not relative_path:
                continue
            
            original_img_path = os.path.join(ROOT, subfolder_prefix, relative_path)
            save_path = os.path.join(ROOT, save_folder_prefix, relative_path)

            if not os.path.exists(original_img_path):
                print(f"경고: 원본 파일을 찾을 수 없습니다. 건너뜁니다: {original_img_path}")
                all_files_saved_successfully = False
                break

            img = cv2.imread(original_img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"경고: 이미지를 읽을 수 없습니다. 건너뜁니다: {original_img_path}")
                all_files_saved_successfully = False
                break

            cropped_img = img[y1:y2, x1:x2]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped_img)
            
            if not os.path.exists(save_path):
                print(f"에러: 이미지 저장에 실패했습니다: {save_path}")
                all_files_saved_successfully = False
                break
        
        if all_files_saved_successfully:
            new_img_info = img_info.copy()
            new_img_info['width'] = new_width
            new_img_info['height'] = new_height
            new_coco_data['images'].append(new_img_info)
            successfully_processed_image_ids.add(img_info['id'])

    # 3. Annotation 필터링 및 좌표 수정
    print(f"\n총 {len(coco_data['annotations'])}개의 Annotation을 필터링 및 수정합니다...")
    
    new_ann_id = 1
    for ann_info in tqdm(coco_data['annotations'], desc="Adjusting annotations"):
        if ann_info['image_id'] not in successfully_processed_image_ids:
            continue

        # 원본 바운딩 박스 좌표 (x, y, 너비, 높이)
        bx, by, bw, bh = ann_info['bbox']
        b_x1, b_y1 = bx, by
        b_x2, b_y2 = bx + bw, by + bh

        # ✨ --- [핵심 개선 로직] --- ✨
        # 1. 자를 영역과 원본 박스의 교집합(겹치는 영역) 계산
        inter_x1 = max(b_x1, x1)
        inter_y1 = max(b_y1, y1)
        inter_x2 = min(b_x2, x2)
        inter_y2 = min(b_y2, y2)

        # 2. 겹치는 영역의 너비와 높이 계산
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1

        # 3. 겹치는 영역이 유효한지(너비와 높이가 0보다 큰지) 확인
        if inter_w > 0 and inter_h > 0:
            # 4. 새로운 좌표계(잘린 이미지 기준)로 변환
            new_bx = inter_x1 - x1
            new_by = inter_y1 - y1

            # 5. 수정된 annotation 정보 생성
            new_ann_info = ann_info.copy()
            new_ann_info['bbox'] = [new_bx, new_by, inter_w, inter_h]
            new_ann_info['id'] = new_ann_id
            
            # (옵션) COCO 형식은 면적(area) 정보도 포함하므로 함께 업데이트
            new_ann_info['area'] = inter_w * inter_h
            
            new_coco_data['annotations'].append(new_ann_info)
            new_ann_id += 1
            
    # 4. 수정된 JSON 파일 저장
    os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)
    print(f"\n성공적으로 처리된 {len(new_coco_data['images'])}개의 이미지 정보와")
    print(f"수정된 {len(new_coco_data['annotations'])}개의 Annotation을 '{SAVE_JSON_PATH}' 파일에 저장합니다.")
    with open(SAVE_JSON_PATH, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

if __name__ == '__main__':
    # 사용자 입력을 통해 설정 가져오기
    config = get_user_input()
    
    if config is not None:
        print("\nStarting crop and filter process...")
        crop_and_filter_coco(config)
        print("\nProcess completed successfully!")
    else:
        print("Process aborted.")