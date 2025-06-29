import json
import os
from pathlib import Path
import shutil

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    COCO bbox [x_min, y_min, width, height] -> YOLO [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, width, height = coco_bbox
    
    # 중심점 계산
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # 정규화 (0~1 범위)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def load_coco_data(json_path):
    """COCO JSON 파일 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_categories(coco_data):
    """카테고리 정보 추출"""
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    category_names = [cat['name'] for cat in coco_data['categories']]
    return categories, category_names

def process_coco_split(coco_data, images_dir, output_dir, split_name, categories):
    """
    단일 COCO 데이터를 YOLO 포맷으로 변환
    """
    # 출력 디렉토리 생성
    img_output_dir = output_dir / 'images' / split_name
    label_output_dir = output_dir / 'labels' / split_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 정보를 딕셔너리로 변환
    images_info = {img['id']: img for img in coco_data['images']}
    
    # 이미지별 annotation 그룹화
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    processed_count = 0
    error_count = 0
    
    # 각 이미지 처리
    for img_id, img_info in images_info.items():
        try:
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # 원본 이미지 파일 경로 찾기
            src_img_path = None
            possible_paths = [
                Path(images_dir) / img_filename,
                Path(images_dir) / Path(img_filename).name,
                Path(images_dir) / split_name / img_filename,
                Path(images_dir) / split_name / Path(img_filename).name
            ]
            
            for path in possible_paths:
                if path.exists():
                    src_img_path = path
                    break
            
            if src_img_path is None:
                print(f"Warning: Image not found for {img_filename}")
                print(f"  Searched paths: {[str(p) for p in possible_paths]}")
                error_count += 1
                continue
            
            # 이미지 파일 복사
            dst_img_path = img_output_dir / Path(img_filename).name
            shutil.copy2(src_img_path, dst_img_path)
            
            # 라벨 파일 생성
            label_filename = Path(img_filename).stem + '.txt'
            label_path = label_output_dir / label_filename
            
            yolo_annotations = []
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    try:
                        if ann['category_id'] not in categories:
                            print(f"Warning: Unknown category_id {ann['category_id']} in {img_filename}")
                            continue
                            
                        category_id = categories[ann['category_id']]  # 0-based로 변환
                        coco_bbox = ann['bbox']
                        
                        # bbox 유효성 검사
                        if len(coco_bbox) != 4 or any(v < 0 for v in coco_bbox):
                            print(f"Warning: Invalid bbox {coco_bbox} in {img_filename}")
                            continue
                        
                        yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                        
                        # 좌표 범위 검사 (0~1)
                        if any(v < 0 or v > 1 for v in yolo_bbox):
                            print(f"Warning: Bbox out of range {yolo_bbox} in {img_filename}")
                            continue
                        
                        # YOLO format: class_id x_center y_center width height
                        yolo_line = f"{category_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}"
                        yolo_annotations.append(yolo_line)
                    except Exception as e:
                        print(f"Error processing annotation in {img_filename}: {e}")
                        continue
            
            # 라벨 파일 저장
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            if len(yolo_annotations) == 0:
                print(f"Info: No valid annotations for {img_filename}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            error_count += 1
            continue
    
    return processed_count, error_count

def convert_coco_to_yolo(train_json_path, test_json_path, images_dir, output_dir):
    """
    COCO format (train.json, test.json)을 YOLO format으로 변환
    
    Args:
        train_json_path: train.json 파일 경로
        test_json_path: test.json 파일 경로  
        images_dir: 이미지 폴더 경로
        output_dir: 출력 디렉토리
    """
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading COCO data...")
    
    # COCO 데이터 로드
    train_data = load_coco_data(train_json_path)
    test_data = load_coco_data(test_json_path)
    
    # 카테고리 정보 추출 및 일관성 확인
    train_categories, train_category_names = extract_categories(train_data)
    test_categories, test_category_names = extract_categories(test_data)
    
    # 카테고리 일관성 확인
    if train_category_names != test_category_names:
        print("Warning: Categories differ between train and test!")
        print(f"Train categories: {train_category_names}")
        print(f"Test categories: {test_category_names}")
        
        # 공통 카테고리 사용
        common_categories = set(train_category_names) & set(test_category_names)
        if not common_categories:
            raise ValueError("No common categories found between train and test!")
        
        category_names = sorted(list(common_categories))
        print(f"Using common categories: {category_names}")
    else:
        category_names = train_category_names
    
    # 최종 카테고리 매핑 (공통 카테고리만)
    final_categories = {i+1: idx for idx, name in enumerate(category_names) for i, cat in enumerate(train_data['categories']) if cat['name'] == name}
    
    print(f"Categories mapping: {final_categories}")
    print(f"Category names: {category_names}")
    
    # Train 데이터 처리
    print(f"\nProcessing training data...")
    print(f"Train images: {len(train_data['images'])}")
    print(f"Train annotations: {len(train_data['annotations'])}")
    
    train_processed, train_errors = process_coco_split(
        train_data, images_dir, output_dir, 'train', train_categories
    )
    
    # Test 데이터 처리 (val로 저장)
    print(f"\nProcessing test data...")
    print(f"Test images: {len(test_data['images'])}")
    print(f"Test annotations: {len(test_data['annotations'])}")
    
    test_processed, test_errors = process_coco_split(
        test_data, images_dir, output_dir, 'val', test_categories
    )
    
    # dataset.yaml 생성
    dataset_yaml = f"""# Custom Dataset Configuration
path: {output_dir.absolute()}
train: images/train
val: images/val
test: # test images (optional)

# Number of classes
nc: {len(category_names)}

# Class names
names: {category_names}

# Original COCO files
# train_json: {train_json_path}
# test_json: {test_json_path}
"""
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"Conversion completed!")
    print(f"{'='*50}")
    print(f"Dataset saved to: {output_dir}")
    print(f"Train images processed: {train_processed} (errors: {train_errors})")
    print(f"Test images processed: {test_processed} (errors: {test_errors})")
    print(f"Total classes: {len(category_names)}")
    print(f"Dataset config: {yaml_path}")
    
    # 구조 출력
    print(f"\nDataset structure:")
    print(f"{output_dir}/")
    print(f"├── images/")
    print(f"│   ├── train/     ({train_processed} images)")
    print(f"│   └── val/       ({test_processed} images)")
    print(f"├── labels/")
    print(f"│   ├── train/     ({train_processed} txt files)")
    print(f"│   └── val/       ({test_processed} txt files)")
    print(f"└── dataset.yaml")
    
    return output_dir

def verify_conversion(output_dir, num_samples=3):
    """변환 결과 검증"""
    output_dir = Path(output_dir)
    
    print(f"\n{'='*50}")
    print(f"Verification Results")
    print(f"{'='*50}")
    
    for split in ['train', 'val']:
        img_dir = output_dir / 'images' / split
        label_dir = output_dir / 'labels' / split
        
        img_files = list(img_dir.glob('*'))
        label_files = list(label_dir.glob('*.txt'))
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(label_files)}")
        
        # 샘플 검증
        for i, img_file in enumerate(img_files[:num_samples]):
            label_file = label_dir / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                print(f"  Sample {i+1}: {img_file.name} -> {len(lines)} objects")
            else:
                print(f"  Sample {i+1}: {img_file.name} -> No label file!")

# 사용 예시
if __name__ == "__main__":
    # 경로 설정
    train_json_path = "/ailab_mat2/dataset/drone/drone_250610/labels/train.json"        # train.json 파일 경로
    test_json_path = "/ailab_mat2/dataset/drone/drone_250610/labels/test.json"          # test.json 파일 경로
    images_dir = "/ailab_mat2/dataset/drone/drone_250610/images"                 # 이미지 폴더 경로
    output_dir = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset"                   # 출력 디렉토리
    
    # 변환 실행
    try:
        result_dir = convert_coco_to_yolo(train_json_path, test_json_path, images_dir, output_dir)
        
        # 결과 검증
        verify_conversion(result_dir)
        
        print(f"\n{'='*50}")
        print(f"Next Steps:")
        print(f"{'='*50}")
        print(f"1. Check the converted dataset:")
        print(f"   ls -la {result_dir}")
        print(f"")
        print(f"2. Start YOLOv5 training:")
        print(f"   python train.py --data {result_dir}/dataset.yaml --weights yolov5m.pt --epochs 100")
        print(f"")
        print(f"3. Verify some samples visually before training!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()