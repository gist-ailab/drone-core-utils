# -*- coding: utf-8 -*-

import os
import json
import datetime
from pathlib import Path

# 경로 설정
input_dir = Path("./ann")
output_dir = Path("./ann_coco")
output_dir.mkdir(parents=True, exist_ok=True)

# Supervisely json 파일 목록 (*.json)
json_files = sorted(input_dir.glob("group_*.json"))

# 전역 category 수집용
global_categories = []

# 먼저 모든 클래스들을 미리 수집 (모든 frame에서 동일한 category ID 유지)
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    for obj in data.get("objects", []):
        class_title = obj["classTitle"]
        if class_title not in global_categories:
            global_categories.append(class_title)

# 전체 annotation ID (고유 ID 유지)
global_ann_id = 1

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    # group 번호 추출
    group_str = json_file.stem.replace(".mp4", "")
    group_num = int(group_str.split("_")[1])

    image_width = data["size"]["width"]
    image_height = data["size"]["height"]

    # object key to class name 매핑
    object_key_to_class = {
        obj["key"]: obj["classTitle"] for obj in data.get("objects", [])
    }

    # 각 frame을 별도 파일로 저장
    for frame in data.get("frames", []):
        frame_idx = frame["index"]
        image_id = group_num * 1000 + frame_idx
        file_name = f"{json_file.stem}_frame_{frame_idx:03d}.png"

        # COCO 형식 기본 구조
        coco_output = {
            "info": {
                "description": f"Frame {frame_idx} from {json_file.name}",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [],
            "images": [{
                "id": image_id,
                "width": image_width,
                "height": image_height,
                "file_name": file_name
            }],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name, "supercategory": "object"}
                for i, name in enumerate(global_categories)
            ]
        }

        for fig in frame.get("figures", []):
            class_name = object_key_to_class.get(fig["objectKey"], "unknown")
            category_id = global_categories.index(class_name) + 1

            pt1, pt2 = fig["geometry"]["points"]["exterior"]
            xmin, ymin = pt1
            xmax, ymax = pt2
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            bbox = [xmin, ymin, width, height]

            annotation = {
                "id": global_ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            global_ann_id += 1

        # frame 별로 저장
        output_name = f"{json_file.stem}_frame_{frame_idx:03d}.json"
        output_path = output_dir / output_name
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=4)

        print(f"[저장 완료] {output_name}")
