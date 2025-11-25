import os
import cv2
import glob
import argparse
import numpy as np



points = []
point_idx = 0
img_copy = None
img = None
resize_factor = 1

def mouse_callback(event, x, y, flags, param):
    """ 마우스 클릭 이벤트로 점을 추가하거나 수정하는 함수 """
    global points, point_idx, img_copy, img, resize_factor


    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭으로 점을 추가
        # 점 추가
        point_idx += 1
        orig_x, orig_y = x, y
        points.append([[orig_x, orig_y]])
        print(f"Point {point_idx} added: ({orig_x}, {orig_y})")
        # 이전 점과 현재 점을 연결하는 선 그리기
        if len(points) > 1:
            prev_x, prev_y = points[-2][0]
            cv2.line(img, (prev_x, prev_y), (orig_x, orig_y), (255, 0, 0), 1)
        
        # 점을 빨간색으로 표시
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(img, str(f'{point_idx}_{round(x/resize_factor),round(y/resize_factor)}'), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow('img', img)
        cv2.imshow('img', cv2.addWeighted(img, 0.2, img_copy, 0.8, 0))
    
    if event == cv2.EVENT_RBUTTONDOWN:  # 우클릭으로 점을 삭제
        if len(points) > 0:
            point_idx -= 1
            points.pop()  # 마지막으로 추가한 점을 삭제
            print("Last point removed.")
            img = img_copy.copy()
            for p in points:  # 남은 점들을 다시 그림
                x, y = p[0]
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.imshow('img', img)

def adjust_point(idx, direction):
    """ 선택된 점의 위치를 ijkl 키로 미세 조정하는 함수 """
    global points, img, img_copy, resize_factor
    
    if len(points) > idx:
        # x, y 좌표를 조정
        x, y = points[idx][0]
        if direction == 'i':  # 위로
            y -= 1
        elif direction == 'j':  # 왼쪽으로
            x -= 1
        elif direction == 'k':  # 아래로
            y += 1
        elif direction == 'l':  # 오른쪽으로
            x += 1
        
        # 좌표 업데이트
        points[idx] = [[x, y]]
        
        # 이미지 복사본에서 점을 다시 그리기
        img = img_copy.copy()
        for p in points:
            x, y = p[0]
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imshow('img', img)



def convert_corners_col_to_row_major(corners, num_cols, num_rows):
    """
    corners: (N, 1, 2) numpy array
    num_cols: 체스보드의 가로 코너 수
    num_rows: 체스보드의 세로 코너 수
    return: 재배열된 corners (row-major order)
    """
    assert corners.shape[0] == num_cols * num_rows, "코너 개수 불일치"

    corners = corners.reshape((num_rows, num_cols, 1, 2))  # 기존: col-major
    reordered = np.transpose(corners, (1, 0, 2, 3))         # transpose rows <-> cols
    reordered = reordered.reshape((-1, 1, 2))               # 다시 (N, 1, 2) 형태로

    return reordered

def draw_corners_on_image(image, corners, radius=3, show_index=True):
    vis_img = image.copy()

    # 무지개 색상 (BGR)
    colors = [
        (0, 0, 255),     # 빨강
        (0, 127, 255),   # 주황
        (0, 255, 255),   # 노랑
        (0, 255, 0),     # 초록
        (255, 0, 0),     # 파랑
        (255, 0, 127),   # 남색
        (255, 0, 255),   # 보라
    ]

    group_size = 8
    for idx, pt in enumerate(corners):
        x, y = int(pt[0][0]), int(pt[0][1])
        
        if idx == 0 or idx == len(corners) - 1:
            color = (255, 255, 255)
        else:
            color = colors[(idx // group_size) % len(colors)]
        cv2.circle(vis_img, (x, y), radius, color, -1)
        if show_index:
            cv2.putText(vis_img, str(idx), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/media/hayeoung/PortableSSD/251120_align')
    parser.add_argument('--cam_name', type=str, default='eo_resize')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=30)
    args = parser.parse_args()

    # 체스보드 설정
    chessboard_size = (6,8)  # (가로 코너 수, 세로 코너 수)
    square_size = 0.035  # 실제 체스보드 한 칸의 크기 (미터 단위, 예: 2.5cm)
    
    # 월드 좌표계에서의 체스보드 좌표 생성
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 모든 이미지에서 2D image points 와 3D object points 저장용
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # 경로 설정
    raw_path = os.path.join(args.root_path, args.cam_name)
    save_path = os.path.join(args.root_path, args.cam_name, "corners")
    os.makedirs(save_path, exist_ok=True)

    # 이미지 파일들 불러오기
    images = os.listdir(raw_path)
    images = [img for img in images if img.endswith('.png')]
    images = [img for img in images if args.start_index <= int(img.split('_')[1].split('.')[0]) and args.start_index <= int(img.split('_')[1].split('.')[0]) <= args.end_index]
    # images.sort()
    
    skip_list = []
 

    resize_factor = 1
    for fname in images:
        img = cv2.imread(os.path.join(raw_path, fname))
        img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        if os.path.exists(os.path.join(save_path, f'corners_{fname.split("_")[1].split(".")[0]}.npy')):
            print(f"You already labeled {fname}.")
            corners = np.load(os.path.join(save_path, f'corners_{fname.split("_")[1].split(".")[0]}.npy'))
            vis_img = draw_corners_on_image(img_copy, corners, show_index=
                                            True)
            cv2.imwrite('corners.png', vis_img)
            _ = input("Press any key to continue...")
            continue

        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # 코너 시각화
            vis_img = draw_corners_on_image(img_copy, corners2, show_index=True)
            cv2.imwrite('corners.png', vis_img)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(50)
            
            reselect_corners = input("Do you want to select corners again? (y/n): ")
            if reselect_corners == 'y':
                img = img.copy()
                ret = False
            else:
                to_row_major = input("Do you want to convert_corners_col_to_row_major the corners? (y/n): ")
                if to_row_major == 'y':
                    corners2 = convert_corners_col_to_row_major(corners2, chessboard_size[0], chessboard_size[1])
                    cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(50)
                
                skip_image = input("Do you want to skip this image? (y/n): ")
                if skip_image == 'y':
                    print(f"Skipping {fname}")
                    skip_list.append(fname)
                    continue
                
                objpoints.append(objp)
                corners2_orig = (corners2 / resize_factor).astype(np.float32)
                imgpoints.append(corners2_orig)
                np.save(os.path.join(save_path, f'corners_{fname.split("_")[1].split(".")[0]}.npy'), corners2_orig)
        
        if not ret:
            print(f"Chessboard corners not found in {fname}")
            img = cv2.addWeighted(img, 0.1, img_copy, 0.9, 0)
            cv2.imshow('img',img)
            cv2.setMouseCallback('img', mouse_callback)
            point_idx = -1
            while True:
                key = cv2.waitKey(0) & 0xFF  # 키 입력 받기
                if key == ord('q'):  # 'q'를 누르면 종료
                    if len(points) != chessboard_size[0] * chessboard_size[1]:
                        print("Please label all points before quitting.")
                    else:
                        print(f"{fname} {len(points)} points labeled.")
                        break
                elif key == ord('i'):  # 'i'를 누르면 첫 번째 점을 위로
                    adjust_point(point_idx, 'i')
                elif key == ord('j'):  # 'j'를 누르면 첫 번째 점을 아래로
                    adjust_point(point_idx, 'j')
                elif key == ord('k'):  # 'k'를 누르면 첫 번째 점을 왼쪽으로
                    adjust_point(point_idx, 'k')
                elif key == ord('l'):  # 'l'을 누르면 첫 번째 점을 오른쪽으로
                    adjust_point(point_idx, 'l')
                
            objpoints.append(objp)
            points = np.array(points)

            points_orig = (np.array(points) / resize_factor)
            imgpoints.append(np.array(points_orig))
            np.save(os.path.join(save_path, f'corners_{fname.split("_")[1].split(".")[0]}.npy'), np.array(points_orig))
            points = []
            point_idx = -1
            
    cv2.destroyAllWindows()

    # 카메라 칼리브레이션 (Intrinsic + Distortion + 각 이미지의 Extrinsic)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, points_orig, gray.shape[::-1], None, None)

    # 결과 출력
    print("\nCamera intrinsic matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)
    print("\nNumber of images for calibration / skipped / total:", len(objpoints), '\t', len(skip_list), '\t', len(images))
    print("\nSkip list:\n", skip_list)