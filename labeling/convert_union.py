import cv2
from pathlib import Path
from natsort import natsorted

input_dir = Path("/media/hayeoung/1f05a827-03e9-4f30-8598-4c826dd0a98f/ailab-drone-01/labeling_250610/labeling_video")
output_dir = input_dir / "merged"
output_dir.mkdir(parents=True, exist_ok=True)

# 파일 이름 기준 정렬
video_files = natsorted(input_dir.glob("*.mp4"))

group = []
group_idx = 0

for vf in video_files:
    # 현재 파일이 `_0.mp4`이면 그룹 마감 & 새로운 그룹 시작
    if "_0.mp4" in vf.name:
        if group:  # 현재까지 모은 그룹이 있다면 병합
            output_path = output_dir / f"group_{group_idx:02d}.mp4"

            # 첫 프레임 크기 추출
            cap = cv2.VideoCapture(str(group[0]))
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading: {group[0]}")
                continue
            height, width = frame.shape[:2]
            cap.release()

            # 비디오 저장 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 20, (width, height))

            # 그룹 내 모든 비디오를 순서대로 병합
            for video_path in group:
                cap = cv2.VideoCapture(str(video_path))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()

            out.release()
            print(f"Saved merged video: {output_path}")
            group_idx += 1

        # 새로운 그룹 시작
        group = [vf]
    else:
        group.append(vf)
