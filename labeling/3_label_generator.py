from pycocotools.coco import COCO
import os
import json
import glob
import random
from tqdm import tqdm

ANN_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/train'
TRAIN_JSON_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/train.json'
TEST_JSON_PATH  = '/ailab_mat2/dataset/drone/drone_250610/labels/test.json'
TRAIN_RATIO = 0.8

def get_grouped_json_files(ann_path):
    """
    ann_path 하위의 모든 json 파일을
    group_name(=json 파일이 있는 디렉토리 이름) 기준으로 묶어서 dict 반환
    """
    group_jsons = {}
    for dirpath, dirnames, filenames in os.walk(ann_path):
        jsons = [os.path.join(dirpath, f) for f in filenames if f.endswith('.json')]
        if jsons:
            group = os.path.basename(dirpath)
            group_jsons[group] = jsons
    return group_jsons

def main():
    # 1) 그룹 단위로 JSON 파일 묶기
    group_jsons = get_grouped_json_files(ANN_PATH)
    groups = list(group_jsons.keys())
    random.shuffle(groups)

    # 2) 그룹 비율에 따라 split
    train_count = int(len(groups) * TRAIN_RATIO)
    train_groups = groups[:train_count]
    test_groups  = groups[train_count:]

    # 3) 각 그룹 안의 파일들을 펼쳐서 최종 리스트 생성
    train_files = [f for g in train_groups for f in group_jsons[g]]
    test_files  = [f for g in test_groups  for f in group_jsons[g]]

    train_annotations = []
    test_annotations  = []
    train_images      = []
    test_images       = []

    annotation_id = 1
    image_id      = 1

    # --- train 파일 처리 ---
    for json_file in tqdm(train_files, desc='Building TRAIN set'):
        group_name = os.path.basename(os.path.dirname(json_file))
        frame_name = os.path.basename(json_file).replace('.json', '.png')

        with open(json_file, 'r') as f:
            data = json.load(f)

        # annotations
        for ann in data['annotations']:
            ann['id']       = annotation_id
            ann['image_id'] = image_id
            train_annotations.append(ann)
            annotation_id += 1

        # images
        for img in data['images']:
            img['id'] = image_id
            img['file_name']  = f'group_rgb/{group_name}/{frame_name}'
            img['depth_path'] = f'group_depth/{group_name}/depth/{frame_name}'
            img['lidar_path'] = f'group_intensity/{group_name}/lidar/{frame_name}'
            img['event_path'] = f'group_ir/{group_name}/event/{frame_name}'
            train_images.append(img)
            image_id += 1

    # --- test 파일 처리 ---
    for json_file in tqdm(test_files, desc='Building TEST set'):
        group_name = os.path.basename(os.path.dirname(json_file))
        frame_name = os.path.basename(json_file).replace('.json', '.png')

        with open(json_file, 'r') as f:
            data = json.load(f)

        for ann in data['annotations']:
            ann['id']       = annotation_id
            ann['image_id'] = image_id
            test_annotations.append(ann)
            annotation_id += 1

        for img in data['images']:
            img['id'] = image_id
            img['file_name']  = f'group_rgb/{group_name}/{frame_name}'
            img['depth_path'] = f'group_depth/{group_name}/depth/{frame_name}'
            img['lidar_path'] = f'group_intensity/{group_name}/lidar/{frame_name}'
            img['event_path'] = f'group_ir/{group_name}/event/{frame_name}'
            test_images.append(img)
            image_id += 1

    # --- COCO 포맷 딕셔너리 작성 ---
    coco_common = {
        "info": {},
        "licenses": [],
        "categories": data['categories'],  # 마지막에 읽힌 data의 categories 사용
    }

    train_coco = {**coco_common, "images": train_images, "annotations": train_annotations}
    test_coco  = {**coco_common, "images": test_images,  "annotations": test_annotations}

    # --- JSON으로 덤프 ---
    with open(TRAIN_JSON_PATH, 'w') as f:
        json.dump(train_coco, f)
    with open(TEST_JSON_PATH, 'w') as f:
        json.dump(test_coco, f)

if __name__ == "__main__":
    main()
