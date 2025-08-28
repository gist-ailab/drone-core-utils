import os
import glob
import cv2
import numpy as np

# --- 경로 설정 (기존과 동일) ---
ROOT = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/images'
RGB_ROOT = os.path.join(ROOT, 'group_rgb')
MODAL_ROOOT = RGB_ROOT.replace('group_rgb', 'group_ir')
SAVE_ROOT = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/heuristic_aligned'

rgb_paths = sorted(glob.glob(os.path.join(RGB_ROOT, '**', '*.png'), recursive=True))

# get_manual_params 함수와 align_image 함수는 이전과 동일하게 사용합니다.
# (코드가 길어 생략)
def get_manual_params(rgb_img, modal_img, initial_params):
    """
    (기존과 동일) 정렬 전/후를 비교하는 화면을 제공합니다.
    """
    params = initial_params.copy()
    window_name = 'Before vs After Alignment (Enter: BATCH SAVE FOLDER, q/e: skip, wasd: move, hj: rot, kl: scale, r:reset, esc:exit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2560, 800)
    h, w = rgb_img.shape[:2]
    center = (w // 2, h // 2)
    action = 'continue'
    while True:
        M = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
        M[0, 2] += params['tx']
        M[1, 2] += params['ty']
        aligned_modal = cv2.warpAffine(modal_img, M, (w, h))
        blended_after = cv2.addWeighted(rgb_img, 0.4, aligned_modal, 0.6, 0)
        info_text = f"tx:{params['tx']}, ty:{params['ty']}, angle:{params['angle']:.1f}, scale:{params['scale']:.2f}"
        cv2.putText(blended_after, "After (Live Aligning)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(blended_after, info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        blended_before = cv2.addWeighted(rgb_img, 0.4, modal_img, 0.6, 0)
        cv2.putText(blended_before, "Before (Original)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        comparison_view = np.hstack((blended_before, blended_after))
        cv2.imshow(window_name, comparison_view)
        key = cv2.waitKeyEx(0)
        if key == 27:
            action = 'quit'
            break
        elif key == 13:
            action = 'save' # 'save'는 이제 폴더 전체 저장을 의미
            break
        elif key == ord('q'):
            action = 'prev'
            break
        elif key == ord('e'):
            action = 'next'
            break
        elif key == ord('r'):
            params = {'tx': 0, 'ty': 0, 'angle': 0.0, 'scale': 1.0}
        elif key == ord('w'): params['ty'] -= 5
        elif key == ord('s'): params['ty'] += 5
        elif key == ord('a'): params['tx'] -= 5
        elif key == ord('d'): params['tx'] += 5
        elif key == ord('h'): params['angle'] -= 0.5
        elif key == ord('j'): params['angle'] += 0.5
        elif key == ord('k'): params['scale'] -= 0.01
        elif key == ord('l'): params['scale'] += 0.01
    cv2.destroyAllWindows()
    return params, action

def align_image(modal_img, params, target_shape):
    """(기존과 동일) 주어진 파라미터로 이미지를 최종 변환합니다."""
    h, w = target_shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
    M[0, 2] += params['tx']
    M[1, 2] += params['ty']
    aligned_modal = cv2.warpAffine(modal_img, M, (w, h))
    return aligned_modal

def main():
    """
    ❗ 메인 로직 수정: 저장 경로에 모달리티 폴더 이름을 포함하도록 변경
    """
    i = 0
    last_params = {'tx': 0, 'ty': 0, 'angle': 0.0, 'scale': 1.0}

    while i < len(rgb_paths):
        # (이전과 동일한 탐색 및 이미지 로드 로직...)
        if i >= len(rgb_paths):
            i = len(rgb_paths) - 1
        rgb_path = rgb_paths[i]
        relative_path = os.path.relpath(rgb_path, RGB_ROOT)
        modal_path = os.path.join(MODAL_ROOOT, relative_path)
        print(f"\n[{i+1}/{len(rgb_paths)}] 탐색 중: {relative_path}")
        if not os.path.exists(modal_path):
            i += 1
            continue
        rgb_img = cv2.imread(rgb_path)
        modal_img = cv2.imread(modal_path)
        if rgb_img is None or modal_img is None:
            i += 1
            continue
        params, action = get_manual_params(rgb_img, modal_img, last_params)

        if action == 'save': # Enter 키를 눌렀을 때
            master_params = params
            
            # ❗ 1. 현재 모달리티 폴더 이름 가져오기
            # 예: MODAL_ROOOT가 '.../group_intensity'라면 'group_intensity'를 가져옴
            modality_folder_name = os.path.basename(MODAL_ROOOT)
            
            # ❗ 2. 최종 저장될 기본 경로를 새로 정의
            final_save_root = os.path.join(SAVE_ROOT, modality_folder_name)

            cv2.destroyAllWindows()
            print("\n" + "="*50)
            # ❗ 3. 사용자 안내 메시지에 새로운 저장 경로 표시
            print(f"현재 파라미터로 모든 이미지를 변환하여 아래 경로에 저장합니다.")
            print(f"▶ 저장 위치: {final_save_root}")
            print(f"▶ 적용 파라미터: {master_params}")
            confirm = input("계속 진행하시겠습니까? (y/n): ")
            print("="*50)

            if confirm.lower() != 'y':
                print("--> 작업을 취소했습니다. 계속 탐색합니다.")
                last_params = params
                continue

            # --- 일괄 처리 시작 ---
            print(f"\n`{MODAL_ROOOT}` 경로의 모든 하위 이미지를 검색합니다...")
            modal_files_to_process = sorted(glob.glob(os.path.join(MODAL_ROOOT, '**', '*.png'), recursive=True))
            total_files = len(modal_files_to_process)
            
            if total_files == 0:
                print("--> 변환할 이미지를 찾지 못했습니다. 프로그램을 종료합니다.")
                break

            print(f"\n총 {total_files}개의 파일에 대해 일괄 변환을 시작합니다...")

            for idx, m_path in enumerate(modal_files_to_process):
                curr_rel_path = os.path.relpath(m_path, MODAL_ROOOT)
                print(f"  [{idx+1}/{total_files}] Processing: {curr_rel_path}")

                r_path = os.path.join(RGB_ROOT, curr_rel_path)
                if not os.path.exists(r_path):
                    print(f"    -> 경고: 짝이 되는 RGB 이미지를 찾을 수 없어 건너뜁니다: {r_path}")
                    continue
                
                m_img = cv2.imread(m_path)
                r_img = cv2.imread(r_path)
                if m_img is None or r_img is None:
                    print("    -> 경고: 이미지 파일을 불러올 수 없어 건너뜁니다.")
                    continue

                aligned = align_image(m_img, master_params, r_img.shape)
                
                # ❗ 4. 새로 정의된 저장 경로(final_save_root)를 사용하여 최종 저장 경로 계산
                s_path = os.path.join(final_save_root, curr_rel_path)
                os.makedirs(os.path.dirname(s_path), exist_ok=True)
                cv2.imwrite(s_path, aligned)

            print("\n--- 일괄 처리가 완료되었습니다. 프로그램을 종료합니다. ---")
            break
            
        elif action == 'next':
            last_params = params
            i += 20
        elif action == 'prev':
            last_params = params
            i = max(0, i - 20)
        elif action == 'quit':
            print("--> 프로그램을 종료합니다.")
            break
            
if __name__ == '__main__':
    main()
    
#적용될 파라미터: {'tx': 45, 'ty': 90, 'angle': -0.5, 'scale': 1.02}