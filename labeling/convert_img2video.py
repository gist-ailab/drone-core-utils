import os
import cv2
from pathlib import Path
from natsort import natsorted

# base 경로 설정
base_path = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610")
output_dir = base_path / "labeling_video"
output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더 없으면 생성

# 모든 drone_* 폴더 탐색
for drone_dir in base_path.glob("drone_*"):
    aligned_dir = drone_dir / "labeling_aligned"
    if not aligned_dir.exists():
        continue

    # labeling_aligned 폴더 내 모든 PNG 파일 가져오기 (정렬 포함)
    image_files = natsorted([f for f in aligned_dir.glob("*.png")])

    if not image_files:
        print(f"No PNG files in {aligned_dir}")
        continue

    # 첫 번째 이미지로 프레임 크기 설정
    first_frame = cv2.imread(str(image_files[0]))
    height, width, _ = first_frame.shape

    # 출력 비디오 파일 이름 설정 (drone_폴더 이름 기준으로)
    output_filename = drone_dir.name + ".mp4"
    output_video_path = output_dir / output_filename

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    print(f"Saved: {output_video_path}")