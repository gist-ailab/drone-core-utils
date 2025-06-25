'''
Label Generator for coco dataset
Glob all annotations for each frame and concat annotations with splitting train, test
maengjemo
'''
from pycocotools.coco import COCO
import os
import json
import os
import glob
import random
from tqdm import tqdm

ANN_PATH= '/ailab_mat2/dataset/drone/drone_250610/labels/train'
TRAIN_JSON_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/train.json'
TEST_JSON_PATH = '/ailab_mat2/dataset/drone/drone_250610/labels/test.json'

def get_all_json_files(ann_path):
    json_files = []
    for dirpath, dirnames, filenames in os.walk(ann_path):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))
    return json_files

def generate_unique_id(existing_ids):
    new_id = random.randint(100000, 999999)
    while new_id in existing_ids:
        new_id = random.randint(100000, 999999)
    return new_id

def main():
    json_files = get_all_json_files(ANN_PATH)
    random.shuffle(json_files)

    train_ratio = 0.8
    train_size = int(len(json_files) * train_ratio)
    train_files = json_files[:train_size]
    test_files = json_files[train_size:]

    train_annotations = []
    test_annotations = []
    train_images = []
    test_images = []

    annotation_id = 1
    image_id = 1

    for json_file in tqdm(train_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for annotation in data['annotations']:
                annotation['id'] = annotation_id
                annotation['image_id'] = image_id
                train_annotations.append(annotation)
                annotation_id += 1
            for image in data['images']:
                image['id'] = image_id
                frame_name = os.path.basename(json_file).replace('.json', '.png')
                group_name = os.path.basename(os.path.dirname(json_file))
                image['file_name'] = f'group_rgb/{group_name}/{frame_name}'
                image['depth_path'] = f'group_depth/{group_name}/depth/{frame_name}'
                image['lidar_path']= f'group_intensity/{group_name}/lidar/{frame_name}'
                image['event_path'] = f'group_ir/{group_name}/event/{frame_name}'

                train_images.append(image)
                image_id += 1

    image_id_offset = image_id
    annotation_id_offset = annotation_id

    for json_file in tqdm(test_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for annotation in data['annotations']:
                annotation['id'] = annotation_id
                annotation['image_id'] = image_id
                test_annotations.append(annotation)
                annotation_id += 1
            for image in data['images']:
                image['id'] = image_id
                image['file_name'] = f'group_rgb/{group_name}/{frame_name}'
                image['depth_path'] = f'group_depth/{group_name}/depth/{frame_name}'
                image['lidar_path']= f'group_intensity/{group_name}/lidar/{frame_name}'
                image['event_path'] = f'group_ir/{group_name}/event/{frame_name}'
                test_images.append(image)
                image_id += 1

    train_coco_format = {
        "info": {},
        "licenses": [],
        "categories": data['categories'],
        "images": train_images,
        "annotations": train_annotations
    }

    test_coco_format = {
        "info": {},
        "licenses": [],
        "categories": data['categories'],
        "images": test_images,
        "annotations": test_annotations
    }

    with open(TRAIN_JSON_PATH, 'w') as f:
        json.dump(train_coco_format, f)

    with open(TEST_JSON_PATH, 'w') as f:
        json.dump(test_coco_format, f)

if __name__ == "__main__":
    main()
