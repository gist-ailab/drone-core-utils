from pathlib import Path

# 기준 경로 설정
base_path = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610")
folders = {
    "depth": base_path / "group_depth",
    "ir": base_path / "group_ir",
    "intensity": base_path / "group_intensity",
    "rgb": base_path / "group_rgb",
    "concat": base_path / "group_concat"
}

# group_* 폴더 기준으로 순회
for group_dir in sorted(folders["depth"].glob("group_*")):
    group_name = group_dir.name

    # 현재 그룹에 해당하는 각 modality 경로
    paths = {k: v / group_name for k, v in folders.items()}

    # 모든 폴더가 존재하는지 확인
    if not all(p.exists() for k, p in paths.items()):
        print(f"[!] Skipping {group_name}: one or more folders missing.")
        continue

    # 각 modality 별 파일 이름 집합 생성
    image_name_sets = {k: {f.name for f in paths[k].glob("*.png")} for k in paths}

    # 전체 union (등장한 모든 파일 이름)
    all_filenames = set.union(*image_name_sets.values())

    for fname in sorted(all_filenames):
        in_concat = fname in image_name_sets["concat"]
        in_others = {k: fname in image_name_sets[k] for k in ["depth", "ir", "intensity", "rgb"]}

        if in_concat:
            # concat에는 있는데 다른 폴더 중 하나라도 없으면 경고
            missing_keys = [k for k, v in in_others.items() if not v]
            if missing_keys:
                print(f"[경고] {group_name}/{fname} → {' '.join(missing_keys)} 폴더에 없음")
        else:
            # concat에는 없고, 다른 폴더 중 하나라도 존재하면 삭제
            existing_keys = [k for k, v in in_others.items() if v]
            if existing_keys:
                print(f"[삭제] {group_name}/{fname} → concat 없음 + {' '.join(existing_keys)} 에 존재")
                for k in existing_keys:
                    f = paths[k] / fname
                    if f.exists():
                        f.unlink()
