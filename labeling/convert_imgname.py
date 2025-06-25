from pathlib import Path
import os
import re
import shutil

# 기준 경로 설정
concat_base = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610/group_concat")
train_base_dir = Path("/home/hayeoung/Downloads/images/train")

# group_00 ~ group_44 순회
for i in range(45):
    group_id = f"{i:02d}"
    concat_dir = concat_base / f"group_{group_id}"
    output_dir = train_base_dir / f"group_{group_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not concat_dir.exists():
        print(f"[스킵] group_{group_id}: concat 폴더 없음")
        continue

    # concat 기준 이미지 정렬
    concat_images = sorted(concat_dir.glob("*.png"))

    # train 폴더에서 해당 group의 이미지 찾기
    train_images = sorted(train_base_dir.glob(f"*_group_{group_id}.mp4_group_{group_id}.mp4_frame_*.png"))

    if not train_images:
        print(f"[스킵] group_{group_id}: train 이미지 없음")
        continue

    for train_img in train_images:
        # frame index 추출
        match = re.search(r"frame_(\d+)\.png", train_img.name)
        if not match:
            print(f"[무시] 이름 형식 오류: {train_img.name}")
            continue

        idx = int(match.group(1))
        if idx >= len(concat_images):
            print(f"[경고] group_{group_id}: index {idx} 초과 (concat에 {len(concat_images)}개)")
            continue

        new_name = concat_images[idx].name
        new_path = output_dir / new_name
        shutil.move(str(train_img), str(new_path))
        print(f"[이동+이름변경] group_{group_id}: {train_img.name} → {new_name}")
