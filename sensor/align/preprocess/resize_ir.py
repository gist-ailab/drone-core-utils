from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import os
import cv2

bridge = CvBridge()

input_folder = "/media/hayeoung/PortableSSD/251120_align/ir_png"
output_folder = "/media/hayeoung/PortableSSD/251120_align/ir_resize"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".png"):
        png_path = os.path.join(input_folder, file_name)
        img = cv2.imread(png_path)

        # -------------------------
        left_strip = img[:, :4]          # shape: (H, 4, 3)
        img_crop = img[:, 4:]            # shape: (H, W-4, 3)
        img_shifted = np.hstack([img_crop, left_strip])
        cropped_img = img_shifted[:-1, 2:]
        img_resized = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
        save_path = os.path.join(output_folder, file_name)
        cv2.imwrite(save_path, img_resized)

        # while True:
        #     cv2.imshow("Image", img_resized)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

print("모든 변환 완료!")