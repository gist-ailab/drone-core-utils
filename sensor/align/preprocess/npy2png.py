from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import os

bridge = CvBridge()

def save_ros_image_to_png(msg, save_path="output.png"):
    # ROS Image → numpy array (RGB로 변환)
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    # PIL로 저장
    pil_img = PILImage.fromarray(img)
    pil_img.save(save_path)

    print(f"✅ Saved: {save_path}")

input_folder = "/media/hayeoung/PortableSSD/251120_align/depth"      # .npy 파일들이 있는 폴더
output_folder = "/media/hayeoung/PortableSSD/251120_align/depth_png"     # 저장될 폴더

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 폴더 안의 모든 파일 반복
for file_name in os.listdir(input_folder):
    if file_name.endswith(".npy"):
        npy_path = os.path.join(input_folder, file_name)

        # npy 파일 로드
        img = np.load(npy_path, allow_pickle=True)
        save_ros_image_to_png(img.item(), os.path.join(output_folder, file_name.replace(".npy", ".png")))
        # print(img)
        # for k, v in img.items():
        #     print(k, type(v))
        # # 값이 0~1이면 0~255로 변환
        # if img.dtype != np.uint8:
        #     img = (img * 255).clip(0, 255).astype(np.uint8)

        # # PNG 파일 이름 설정
        # png_name = file_name.replace(".npy", ".png")
        # png_path = os.path.join(output_folder, png_name)

        # # 저장
        # Image.fromarray(img).save(png_path)
        # print(f"Saved: {png_path}")

print("모든 변환 완료!")

# import cv2

# img_path = "/media/hayeoung/PortableSSD/251120_align/eo_png/eo_0.png"
# img = cv2.imread(img_path)

# while True:
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()