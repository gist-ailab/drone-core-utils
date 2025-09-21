import os
import json
from tqdm import tqdm

def get_user_input():
    """
    사용자로부터 설정 정보를 입력받는 함수
    """
    print("="*60)
    print("ANNOTATION FILTER TOOL")
    print("Filter annotations for non-existing images")
    print("="*60)
    
    # 1. JSON 파일 경로
    json_path = input("Enter JSON file path: ").strip()
    if not json_path or not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return None
    
    # 2. 이미지 루트 디렉토리
    image_root = input("Enter image root directory: ").strip()
    if not image_root or not os.path.exists(image_root):
        print(f"Image root directory not found: {image_root}")
        return None
    
    # 3. 출력 JSON 파일 경로
    default_output = json_path.replace('.json', '_filtered.json')
    output_path = input(f"Enter output JSON file path\n(default: {default_output})\n> ").strip()
    if not output_path:
        output_path = default_output
    
    config = {
        'json_path': json_path,
        'image_root': image_root,
        'output_path': output_path
    }
    
    # 설정 확인
    print("\n" + "-"*60)
    print("CONFIGURATION SUMMARY:")
    print("-"*60)
    print(f"Input JSON: {json_path}")
    print(f"Image root: {image_root}")
    print(f"Output JSON: {output_path}")
    print("-"*60)
    
    confirm = input("Proceed with these settings? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return None
    
    return config

def check_image_exists(image_info, image_root):
    """
    이미지 파일이 존재하는지 확인하는 함수
    """
    # 여러 가능한 경로 확인
    possible_paths = []
    
    # 1. file_name 직접 사용
    if 'file_name' in image_info and image_info['file_name']:
        possible_paths.append(os.path.join(image_root, image_info['file_name']))
    
    # 2. 다른 경로 필드들 확인
    path_fields = ['depth_path', 'lidar_path', 'event_path', 'thermal_path', 'ir_path']
    for field in path_fields:
        if field in image_info and image_info[field]:
            possible_paths.append(os.path.join(image_root, image_info[field]))
    
    # 3. id를 기반으로 한 파일명 추정 (필요시)
    if 'id' in image_info:
        # 일반적인 파일 확장자들로 시도
        extensions = ['.png', '.jpg', '.jpeg']
        for ext in extensions:
            possible_paths.append(os.path.join(image_root, f"{image_info['id']}{ext}"))
    
    # 존재하는 파일 찾기
    existing_paths = []
    for path in possible_paths:
        if os.path.exists(path):
            existing_paths.append(path)
    
    return len(existing_paths) > 0, existing_paths

def filter_annotations(config):
    """
    존재하지 않는 이미지에 대한 annotation을 필터링하는 함수
    """
    json_path = config['json_path']
    image_root = config['image_root']
    output_path = config['output_path']
    
    print(f"\nLoading JSON file: {json_path}")
    
    # JSON 파일 로드
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Original dataset:")
    print(f"  - Images: {len(coco_data.get('images', []))}")
    print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
    
    # 새로운 COCO 데이터 준비
    filtered_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', []),
        'images': [],
        'annotations': []
    }
    
    # 존재하는 이미지들의 ID 수집
    valid_image_ids = set()
    missing_images = []
    
    print(f"\nChecking image files...")
    
    for img_info in tqdm(coco_data.get('images', []), desc="Checking images"):
        exists, existing_paths = check_image_exists(img_info, image_root)
        
        if exists:
            # 이미지가 존재하면 유효한 이미지로 추가
            filtered_data['images'].append(img_info)
            valid_image_ids.add(img_info['id'])
        else:
            # 이미지가 존재하지 않으면 누락된 이미지로 기록
            missing_images.append({
                'id': img_info['id'],
                'file_name': img_info.get('file_name', 'N/A'),
                'searched_paths': [os.path.join(image_root, img_info.get('file_name', ''))]
            })
    
    print(f"\nImage filtering results:")
    print(f"  - Valid images: {len(valid_image_ids)}")
    print(f"  - Missing images: {len(missing_images)}")
    
    # 누락된 이미지 정보 출력 (처음 10개만)
    if missing_images:
        print(f"\nFirst {min(10, len(missing_images))} missing images:")
        for i, missing in enumerate(missing_images[:10]):
            print(f"  {i+1}. ID: {missing['id']}, File: {missing['file_name']}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    # 유효한 이미지에 대한 annotation만 필터링
    print(f"\nFiltering annotations...")
    
    valid_annotations = 0
    removed_annotations = 0
    
    for ann_info in tqdm(coco_data.get('annotations', []), desc="Filtering annotations"):
        if ann_info['image_id'] in valid_image_ids:
            filtered_data['annotations'].append(ann_info)
            valid_annotations += 1
        else:
            removed_annotations += 1
    
    print(f"\nAnnotation filtering results:")
    print(f"  - Valid annotations: {valid_annotations}")
    print(f"  - Removed annotations: {removed_annotations}")
    
    # 결과 저장
    print(f"\nSaving filtered dataset to: {output_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # 최종 요약
    print(f"\n" + "="*60)
    print("FILTERING SUMMARY:")
    print("="*60)
    print(f"Original images: {len(coco_data.get('images', []))}")
    print(f"Filtered images: {len(filtered_data['images'])}")
    print(f"Removed images: {len(coco_data.get('images', [])) - len(filtered_data['images'])}")
    print()
    print(f"Original annotations: {len(coco_data.get('annotations', []))}")
    print(f"Filtered annotations: {len(filtered_data['annotations'])}")
    print(f"Removed annotations: {len(coco_data.get('annotations', [])) - len(filtered_data['annotations'])}")
    print()
    print(f"Output file: {output_path}")
    print("="*60)
    
    # 누락된 이미지 리스트를 별도 파일로 저장
    if missing_images:
        missing_file = output_path.replace('.json', '_missing_images.txt')
        with open(missing_file, 'w') as f:
            f.write("Missing Images Report\n")
            f.write("="*50 + "\n\n")
            for missing in missing_images:
                f.write(f"ID: {missing['id']}\n")
                f.write(f"File: {missing['file_name']}\n")
                f.write(f"Searched: {missing['searched_paths'][0]}\n")
                f.write("-" * 30 + "\n")
        
        print(f"Missing images list saved to: {missing_file}")

def main():
    """메인 함수"""
    try:
        config = get_user_input()
        
        if config is not None:
            print("\nStarting annotation filtering...")
            filter_annotations(config)
            print("\nFiltering completed successfully!")
        else:
            print("Process aborted.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
