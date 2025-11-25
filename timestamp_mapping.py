#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
import tf.transformations as tft
import sensor_msgs.point_cloud2 as pc2
import rospy

# ==============================================================================
# 설정 (Configuration)
# ==============================================================================
CONFIG = {
    'bag_file': '../concept-graphs/minsangbeom_1_2025-07-08-22-24-21.bag',
    
    # 처리할 rosbag의 비율 (1.0 = 100%, 0.7 = 70%)
    'process_duration_ratio': 0.5,

    'lidar_topic': '/ml_/pointcloud',
    'rgb_topic': '/EO/EO/camera/image',
    'ir_topic': '/IR/IR/camera/image',
    'ambient_topic': '/ml_/ambient_color',
    'intensity_topic': '/ml_/intensity_color',
    'depth_topic': '/ml_/depth_color',

    'sync_tolerance_sec': 0.05,
    'map_frame': 'map',
}

def find_closest_message(target_stamp, messages):
    min_diff = float('inf')
    closest_msg = None
    for stamp, msg in messages:
        diff = abs(target_stamp - stamp)
        if diff < min_diff:
            min_diff = diff
            closest_msg = msg
    
    if min_diff > CONFIG['sync_tolerance_sec']:
        return None
    return closest_msg

def main():
    print("[1] rosbag 파일에서 데이터 추출 시작...")
    
    lidar, rgb, ir = [], [], []
    bridge = CvBridge()
    
    topics_to_read = [CONFIG['lidar_topic'], CONFIG['rgb_topic'], CONFIG['ir_topic']]
    
    with rosbag.Bag(CONFIG['bag_file'], 'r') as bag:
        start_time_ros = bag.get_start_time()
        end_time_ros = bag.get_end_time()
        duration = end_time_ros - start_time_ros
        cutoff_time_ros = start_time_ros + duration * CONFIG['process_duration_ratio']
        
        print(f"Rosbag 전체 시간: {duration:.2f}초")
        print(f"-> 처리할 시간: {duration * CONFIG['process_duration_ratio']:.2f}초 (앞부분 {CONFIG['process_duration_ratio']*100:.0f}%)")

        # 시간 필터링 없이, 지정된 토픽의 전체 메시지 수를 계산
        message_count_for_pbar = bag.get_message_count(topic_filters=topics_to_read)
        pbar = tqdm(total=message_count_for_pbar, unit="msg")

        # 메시지를 읽을 때는 시간 필터링을 그대로 유지합니다.
        for topic, msg, t in bag.read_messages(topics=topics_to_read, start_time=rospy.Time.from_sec(start_time_ros), end_time=rospy.Time.from_sec(cutoff_time_ros)):
            pbar.update(1)
            if not hasattr(msg, 'header'): continue
            timestamp = msg.header.stamp.to_sec()
            if topic == CONFIG['lidar_topic']:
                lidar.append((timestamp, msg))
            elif topic == CONFIG['rgb_topic']:
                try:
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg) if 'compressed' in topic else bridge.imgmsg_to_cv2(msg, "bgr8")
                    rgb.append((timestamp, cv_image))
                except Exception: pass
            elif topic == CONFIG['ir_topic']:
                ir.append((timestamp, msg))
        pbar.close()

    print(f"\n데이터 추출 완료: LiDAR 스캔 {len(lidar)}개, 이미지 {len(rgb)}개, Odometry 메시지 {len(ir)}개")
    print("[2] 타임스탬프 매칭 확인 시작...")

    matching_count = 0
    for lidar_stamp, lidar_msg in tqdm(lidar, unit="scan"):
        
        closest_image = find_closest_message(lidar_stamp, rgb)
        if closest_image is None: continue
        
        closest_odom_msg = find_closest_message(lidar_stamp, ir)
        if closest_odom_msg is not None:
            matching_count += 1
        else:
            continue

    print(f"a number of timestamp matching frames : {matching_count}")

if __name__ == '__main__':
    main()