import shutil
from pathlib import Path
from natsort import natsorted

# 기본 경로 설정
base_dir = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610")
output_base = base_dir / "group_rgb"
output_base.mkdir(parents=True, exist_ok=True)

# 모든 drone_* 폴더 정렬해서 가져오기
drone_folders = natsorted([p for p in base_dir.glob("drone_*") if p.is_dir()])

group = []
group_idx = 0

for folder in drone_folders:
    if folder.name.endswith("_0"):
        if group:  # 현재까지 모인 그룹을 처리
            group_output_dir = output_base / f"group_{group_idx:02d}"
            group_output_dir.mkdir(parents=True, exist_ok=True)

            for drone in group:
                image_dir = drone / "rgb_aligned"
                if not image_dir.exists():
                    continue

                for img_path in natsorted(image_dir.glob("*.png")):
                    shutil.copy(img_path, group_output_dir / img_path.name)

            print(f"Copied group {group_idx:02d} to {group_output_dir}")
            group_idx += 1

        # 새로운 그룹 시작
        group = [folder]
    else:
        group.append(folder)

# 마지막 그룹 처리 (loop 종료 후 남아있는 것)
if group:
    group_output_dir = output_base / f"group_{group_idx:02d}"
    group_output_dir.mkdir(parents=True, exist_ok=True)

    for drone in group:
        image_dir = drone / "rgb_aligned"
        if not image_dir.exists():
            continue

        for img_path in natsorted(image_dir.glob("*.png")):
            shutil.copy(img_path, group_output_dir / img_path.name)

    print(f"Copied group {group_idx:02d} to {group_output_dir}")
