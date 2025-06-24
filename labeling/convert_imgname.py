from pathlib import Path
import os
import re
import shutil

# 공통 기준 경로
concat_base = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610/group_concat")

# 처리 타입 지정: "image" 또는 "label"
mode = "both"  # "image", "label", 또는 "both"

# 각 타입별 설정
targets = []

if mode in ("image", "both"):
    targets.append({
        "type": "image",
        "source_dir": Path("/home/hayeoung/Downloads/images/train"),
        "output_base": Path("/home/hayeoung/Downloads/images/train"),
        "pattern": "*_group_{gid}.mp4_group_{gid}.mp4_frame_*.png",
        "match_re": r"frame_(\d+)\.png",
        "new_ext": ".png"
    })

if mode in ("label", "both"):
    targets.append({
        "type": "label",
        "source_dir": Path("/home/hayeoung/Downloads/labels/train"),
        "output_base": Path("/home/hayeoung/Downloads/labels/train"),
        "pattern": "*_group_{gid}.mp4_group_{gid}.mp4_frame_*.txt",
        "match_re": r"frame_(\d+)\.txt",
        "new_ext": ".txt"
    })

# group_00 ~ group_44 순회
for i in range(45):
    group_id = f"{i:02d}"
    concat_dir = concat_base / f"group_{group_id}"

    if not concat_dir.exists():
        print(f"[스킵] group_{group_id}: concat 폴더 없음")
        continue

    # concat 기준 이미지 목록
    concat_images = sorted(concat_dir.glob("*.png"))

    for target in targets:
        source_dir = target["source_dir"]
        output_dir = target["output_base"] / f"group_{group_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        pattern = target["pattern"].format(gid=group_id)
        match_re = target["match_re"]
        new_ext = target["new_ext"]
        file_type = target["type"]

        # 파일 수집
        files = sorted(source_dir.glob(pattern))
        if not files:
            print(f"[스킵] group_{group_id}: {file_type} 파일 없음")
            continue

        for f in files:
            match = re.search(match_re, f.name)
            if not match:
                print(f"[무시] 이름 형식 오류: {f.name}")
                continue

            idx = int(match.group(1))
            if idx >= len(concat_images):
                print(f"[경고] group_{group_id}: index {idx} 초과 (concat에 {len(concat_images)}개)")
                continue

            # 기준 이름에서 .png → .txt 등으로 변경
            new_name = concat_images[idx].stem + new_ext
            new_path = output_dir / new_name
            shutil.move(str(f), str(new_path))
            print(f"[이동+이름변경] {file_type} group_{group_id}: {f.name} → {new_name}")
