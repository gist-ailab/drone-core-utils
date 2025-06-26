from pycocotools.coco import COCO
import os
import json
import random
from tqdm import tqdm

IMG_ROOT = '/ailab_mat2/dataset/drone/drone_250610/images'
ANN_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/train'
TRAIN_JSON_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/train.json'
TEST_JSON_PATH  = '/ailab_mat2/dataset/drone/drone_250610/labels/test.json'
TRAIN_RATIO = 0.8

def get_grouped_json_files(ann_path):
    group_jsons = {}
    for dirpath, dirnames, filenames in os.walk(ann_path):
        jsons = [os.path.join(dirpath, f) for f in filenames if f.endswith('.json')]
        if jsons:
            group = os.path.basename(dirpath)
            group_jsons[group] = jsons
    return group_jsons

def adjust_bbox(bbox):
    """
    bbox: [x, y, w, h] in the 960-wide concat image.
    Returns a new [x, y, w, h] in a single 480-wide image,
    clamped/translated as described.
    """
    x, y, w, h = bbox
    x2 = x + w

    # 1) 완전 RGB 영역
    if x2 <= 480:
        return [x, y, w, h]

    # 2) RGB→IR 경계 넘긴 박스: 오른쪽 끝을 480에 클램핑
    if x < 480 < x2:
        new_w = 480 - x
        return [x, y, new_w, h]

    # 3) 완전 IR 영역
    if x >= 480:
        new_x = x - 480
        return [new_x, y, w, h]

    # (이외의 경우, e.g. 완전히 밖으로 벗어나면 None 처리)
    return None

def main():
    group_jsons = get_grouped_json_files(ANN_PATH)
    groups = list(group_jsons.keys())
    random.shuffle(groups)

    train_count = int(len(groups) * TRAIN_RATIO)
    train_groups = groups[:train_count]
    test_groups  = groups[train_count:]

    train_files = [f for g in train_groups for f in group_jsons[g]]
    test_files  = [f for g in test_groups  for f in group_jsons[g]]

    train_annotations, test_annotations = [], []
    train_images,      test_images      = [], []

    annotation_id = 1
    image_id      = 1

    def has_all_files(img_dict, group_name):
        frame = os.path.basename(img_dict['file_name'])
        paths = [
            os.path.join(IMG_ROOT, 'group_rgb',      group_name, frame),
            os.path.join(IMG_ROOT, 'group_depth',    group_name, frame),
            os.path.join(IMG_ROOT, 'group_intensity',group_name, frame),
            os.path.join(IMG_ROOT, 'group_ir',       group_name, frame),
        ]
        return all(os.path.exists(p) for p in paths)

    # --- TRAIN ---
    for json_file in tqdm(train_files, desc='Building TRAIN set'):
        group_name = os.path.basename(os.path.dirname(json_file))
        frame_name = os.path.basename(json_file).replace('.json', '.png')
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not all(has_all_files(img, group_name) for img in data['images']):
            tqdm.write(f"[SKIP] Missing files for {json_file}")
            continue

        for ann in data['annotations']:
            new_bbox = adjust_bbox(ann['bbox'])
            if new_bbox is None:
                continue  # (원한다면 완전히 벗어난 박스는 버림)
            ann['bbox']       = new_bbox
            ann['id']         = annotation_id
            ann['image_id']   = image_id
            train_annotations.append(ann)
            annotation_id   += 1

        for img in data['images']:
            img['id'] = image_id
            img['file_name']  = f'group_rgb/{group_name}/{frame_name}'
            img['depth_path'] = f'group_depth/{group_name}/{frame_name}'
            img['lidar_path'] = f'group_intensity/{group_name}/{frame_name}'
            img['event_path'] = f'group_ir/{group_name}/{frame_name}'
            train_images.append(img)
            image_id += 1

    # --- TEST ---
    for json_file in tqdm(test_files, desc='Building TEST set'):
        group_name = os.path.basename(os.path.dirname(json_file))
        frame_name = os.path.basename(json_file).replace('.json', '.png')
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not all(has_all_files(img, group_name) for img in data['images']):
            tqdm.write(f"[SKIP] Missing files for {json_file}")
            continue

        for ann in data['annotations']:
            new_bbox = adjust_bbox(ann['bbox'])
            if new_bbox is None:
                continue
            ann['bbox']       = new_bbox
            ann['id']         = annotation_id
            ann['image_id']   = image_id
            test_annotations.append(ann)
            annotation_id   += 1

        for img in data['images']:
            img['id'] = image_id
            img['file_name']  = f'group_rgb/{group_name}/{frame_name}'
            img['depth_path'] = f'group_depth/{group_name}/{frame_name}'
            img['lidar_path'] = f'group_intensity/{group_name}/{frame_name}'
            img['event_path'] = f'group_ir/{group_name}/{frame_name}'
            test_images.append(img)
            image_id += 1

    coco_common = {
        "info": {},
        "licenses": [],
        "categories": data['categories'],
    }
    train_coco = {**coco_common, "images": train_images, "annotations": train_annotations}
    test_coco  = {**coco_common, "images": test_images,  "annotations": test_annotations}

    with open(TRAIN_JSON_PATH, 'w') as f:
        json.dump(train_coco, f)
    with open(TEST_JSON_PATH, 'w') as f:
        json.dump(test_coco, f)

if __name__ == "__main__":
    main()
