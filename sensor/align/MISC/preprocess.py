import cv2

img_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/rgb.png'
depth_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/depth.png'
intensity_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/sensors/align_0524/intensity.png'

img = cv2.imread(img_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
intensity = cv2.imread(intensity_path, cv2.IMREAD_UNCHANGED)


rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90도 시계 방향 회전
rotated_depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)  # 90도 시계 방향 회전
rotate_intensity = cv2.rotate(intensity, cv2.ROTATE_90_CLOCKWISE)  # 90도 시계 방향 회전

cv2.imwrite('rotated_img.png', rotated_img)
cv2.imwrite('rotated_depth.png', rotated_depth)
cv2.imwrite('rotated_intensity.png', rotate_intensity)