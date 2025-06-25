from pathlib import Path
import os
import re
import shutil

# 기준 경로
concat_base = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610/group_concat")
label_base_dir = Path("/home/hayeoung/Downloads/drone_250610/dataset 2025-06-14 08-09-41/ann_coco")
# train_base_dir = Path("/home/hayeoung/Downloads/images/train")

# group_00 ~ group_44 순회
for i in range(45):
    group_id = f"{i:02d}"
    concat_dir = concat_base / f"group_{group_id}"
    output_dir = label_base_dir / f"group_{group_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not concat_dir.exists():
        print(f"[스킵] group_{group_id}: concat 폴더 없음")
        continue

    # concat 이미지 기준 정렬 (이름만 사용)
    concat_images = sorted(concat_dir.glob("*.png"))

    # 해당 group의 라벨(txt) 파일 찾기
    label_files = sorted(label_base_dir.glob(f"group_{group_id}.mp4_frame_*.json"))

    if not label_files:
        print(f"[스킵] group_{group_id}: 라벨 없음")
        continue

    for label_path in label_files:
        match = re.search(r"frame_(\d+)\.json", label_path.name)
        if not match:
            print(f"[무시] 이름 형식 오류: {label_path.name}")
            continue

        idx = int(match.group(1))
        if idx >= len(concat_images):
            print(f"[경고] group_{group_id}: index {idx} 초과 (concat에 {len(concat_images)}개)")
            continue

        # 기준 이미지 이름에서 .png → .txt
        new_name = concat_images[idx].stem + ".json"
        new_path = output_dir / new_name
        shutil.move(str(label_path), str(new_path))
        print(f"[이동+이름변경] group_{group_id}: {label_path.name} → {new_name}")