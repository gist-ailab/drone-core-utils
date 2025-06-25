# Drone Labeling Scripts
---

## 📁 구성 스크립트

### 1. `move_files.py`

#### 📌 기능
- `drone_*` 폴더의 번호를 기준으로 그룹화
- 각 그룹별로 다음과 같은 폴더 생성 후 이미지 이동:
  - `group_depth/`
  - `group_intensity/`
  - `group_ir/`
  - `group_rgb/`
  - `group_concat/`

#### 🧪 예시
- `drone_*_03`, `drone_*_04` → `group_01`
  - 생성된 폴더 예시:
    - `group_rgb/group_01/`
    - `group_concat/group_01/`

---

### 2. `convert_labelname.py`

#### 📌 기능
- `group_concat` 폴더 내 라벨링 `.json` 파일들을 정렬
- 기존 라벨 파일명을 원본 그룹 이미지 파일 이름과 장랼 슨사러 정확히 매칭하여 변경

#### 📂 파일명 매칭 예시
- 기존: _group_0.mp4_group_0.mp4_frame_3.json (0번 그룹 4번째(3번) 이미지)
- 변경 후: frame_1741589575.571219.json (← group_00/frame_1741589575.571219.png와 매칭)

---

### 3. `convert_json.py`

#### 📌 기능
- `.json` 라벨 파일 내 메타데이터 수정:
- `"description"` 필드 업데이트
- `"file_name"` 필드 값을 실제 이미지 이름에 맞게 수정

#### 📂 변경 예시
```json
{
"description": "Converted from original label",
"file_name": "frame_1741589575.571219.png",
}
