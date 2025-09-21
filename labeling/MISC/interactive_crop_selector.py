import cv2
import os
import numpy as np

class CropSelector:
    def __init__(self, image_path, lidar_path):
        """
        이미지에서 인터랙티브하게 crop 영역을 선택하는 클래스
        """
        self.image_path = image_path
        self.original_img = None
        self.display_img = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.crop_coords = None
        self.lidar_path = lidar_path
        
        # 이미지 로드
        self.load_image()
        
    def load_image(self):
        """이미지를 로드하고 표시용 이미지를 준비"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        self.original_img = cv2.imread(self.image_path)
        self.original_lidar = cv2.imread(self.lidar_path)
        self.original_img = cv2.addWeighted(self.original_img, 0.5, self.original_lidar, 0.5, 0)
        if self.original_img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
            
        self.display_img = self.original_img.copy()
        print(f"Loaded image: {self.image_path}")
        print(f"Image size: {self.original_img.shape[1]}x{self.original_img.shape[0]}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 콜백 함수"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 마우스 왼쪽 버튼 클릭 시작
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 마우스 이동 중 (드래그)
            if self.drawing:
                self.end_point = (x, y)
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 마우스 왼쪽 버튼 떼기
            self.drawing = False
            self.end_point = (x, y)
            self.finalize_selection()
            
    def update_display(self):
        """현재 선택 중인 사각형을 표시"""
        self.display_img = self.original_img.copy()
        
        if self.start_point and self.end_point:
            # 현재 드래그 중인 사각형 그리기
            cv2.rectangle(self.display_img, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # 좌표 정보 표시
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # 좌표를 정규화 (x1 < x2, y1 < y2)
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            width = max_x - min_x
            height = max_y - min_y
            
            info_text = f"({min_x}, {min_y}) -> ({max_x}, {max_y}) | {width}x{height}"
            cv2.putText(self.display_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Crop Selector', self.display_img)
        
    def finalize_selection(self):
        """선택 완료 후 최종 좌표 계산"""
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # 좌표를 정규화 (x1 < x2, y1 < y2)
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            self.crop_coords = {
                'x1': min_x,
                'y1': min_y, 
                'x2': max_x,
                'y2': max_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            }
            
            # 최종 사각형 그리기
            self.display_img = self.original_img.copy()
            cv2.rectangle(self.display_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
            
            # 최종 정보 표시
            info_text = f"FINAL: ({min_x}, {min_y}) -> ({max_x}, {max_y}) | {max_x-min_x}x{max_y-min_y}"
            cv2.putText(self.display_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.display_img, "Press 's' to save, 'r' to reset, 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Crop Selector', self.display_img)
            
    def run(self):
        """메인 실행 함수"""
        window_name = 'Crop Selector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nInstructions:")
        print("- Click and drag to select crop area")
        print("- Press 's' to save coordinates")
        print("- Press 'r' to reset selection")
        print("- Press 'q' to quit")
        
        # 초기 이미지 표시
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # 선택 리셋
                self.start_point = None
                self.end_point = None
                self.crop_coords = None
                self.display_img = self.original_img.copy()
                cv2.imshow(window_name, self.display_img)
                print("Selection reset")
            elif key == ord('s'):
                # 좌표 저장
                if self.crop_coords:
                    self.save_coordinates()
                else:
                    print("No selection made yet!")
                    
        cv2.destroyAllWindows()
        return self.crop_coords
        
    def save_coordinates(self):
        """좌표를 파일에 저장하고 99_crop_dataset.py에서 사용할 수 있는 형태로 출력"""
        if not self.crop_coords:
            print("No coordinates to save!")
            return
            
        coords = self.crop_coords
        
        print("\n" + "="*50)
        print("CROP COORDINATES SELECTED:")
        print("="*50)
        print(f"Top-left: ({coords['x1']}, {coords['y1']})")
        print(f"Bottom-right: ({coords['x2']}, {coords['y2']})")
        print(f"Width: {coords['width']}")
        print(f"Height: {coords['height']}")
        print("\n" + "-"*50)
        print("FOR 99_crop_dataset.py, use these coordinates:")
        print("-"*50)
        print(f"x1, y1 = {coords['x1']}, {coords['y1']}")
        print(f"x2, y2 = {coords['x2']}, {coords['y2']}")
        print("-"*50)
        
        # 좌표를 텍스트 파일로도 저장
        output_file = "crop_coordinates.txt"
        with open(output_file, 'w') as f:
            f.write(f"# Crop coordinates for {os.path.basename(self.image_path)}\n")
            f.write(f"# Image size: {self.original_img.shape[1]}x{self.original_img.shape[0]}\n")
            f.write(f"x1, y1 = {coords['x1']}, {coords['y1']}\n")
            f.write(f"x2, y2 = {coords['x2']}, {coords['y2']}\n")
            f.write(f"# Crop size: {coords['width']}x{coords['height']}\n")
            
        print(f"\nCoordinates saved to: {output_file}")
        
def main():
    """메인 함수"""
    # 기본 이미지 경로 (사용자가 지정한 이미지)
    default_ir_path = "/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/250312_sejong/drone_250312_sejong_multimodal_coco_synced/images_heuristic/group_ir/group_00/frame_1741589576.111370.png"
    default_lidar_path = default_ir_path.replace('group_ir', 'group_intensity')
    # 이미지 경로 입력 받기
    image_path = input(f"Enter image path (or press Enter for default):\n{default_ir_path}\n> ").strip()
    lidar_path = input(f"Enter lidar path (or press Enter for default):\n{default_lidar_path}\n> ").strip()
    if not image_path:
        image_path = default_ir_path
        lidar_path = default_lidar_path
    try:
        # CropSelector 실행
        selector = CropSelector(image_path, lidar_path)
        coordinates = selector.run()
        
        if coordinates:
            print(f"\nFinal coordinates: {coordinates}")
        else:
            print("No coordinates selected.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
