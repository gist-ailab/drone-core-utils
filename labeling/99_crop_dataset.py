import os
import json
import cv2
from tqdm import tqdm

# --- 설정 (Configuration) ---
# 자를 영역 좌표
x1, y1 = 40, 110
x2, y2 = 480, 480

ROOT = '/media/ailab/SSD2/Workspace/dst/drone_250312_sejong_multimodal_coco_cropped'
subfolder_prefix = 'images'
save_folder_prefix = 'images_cropped3' # 저장할 폴더 이름
JSON_PATH = os.path.join(ROOT, 'labels', 'train.json')
SAVE_JSON_PATH = os.path.join(ROOT, 'labels', 'train_cropped3.json')


def crop_and_filter_coco():
    """
    COCO 데이터셋의 이미지를 자르고, 그에 맞게 Annotation을 필터링 및 수정합니다.
    잘린 영역에 걸치는 바운딩 박스는 좌표를 재계산합니다.
    """
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
    crop_and_filter_coco()