import rosbag
import sys

# --- 사용자 설정 ---
BAG_FILE_PATH = "/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/DATA/0310/drone_2025-03-10-15-54-11_7.bag"

topics_to_check = {
    "/rgb/image_raw": "RGB",
    "/ilidar/depth": "LiDAR Depth",
    "/ilidar/intensity": "LiDAR Intensity",
    "/ir/image_raw": "IR"
}

# --------------------

def check_timestamps():
    """
    Bag 파일의 각 토픽에서 메시지를 읽고,
    메시지 헤더의 타임스탬프가 유효한지 확인하고 출력합니다.
    """
    try:
        bag = rosbag.Bag(BAG_FILE_PATH, 'r')
    except rosbag.BagError as e:
        print(f"Error: Could not open bag file. Please check the path. {e}", file=sys.stderr)
        return

    print("--- Analyzing Message Timestamps ---")
    
    first_stamps = {}
    for topic, _ in topics_to_check.items():
        first_stamps[topic] = None

    # 모든 메시지를 순회하며 첫 타임스탬프를 찾고, 메시지 헤더 유효성 검사
    for topic, msg, _ in bag.read_messages(topics=topics_to_check.keys()):
        if topic in first_stamps and first_stamps[topic] is None:
            try:
                # header.stamp가 존재하는지 확인
                stamp = msg.header.stamp.to_sec()
                first_stamps[topic] = stamp
            except AttributeError:
                print(f"Error: Topic '{topic}' has no valid header.stamp field.", file=sys.stderr)
                first_stamps[topic] = "Invalid"
    
    bag.close()
    
    # 결과 출력
    for topic, name in topics_to_check.items():
        stamp = first_stamps.get(topic)
        if stamp == "Invalid":
            print(f"[{name}]: No valid timestamp found.")
        elif stamp is not None:
            print(f"[{name}]: First timestamp is {stamp:.6f}")
        else:
            print(f"[{name}]: No messages found for this topic.")
            
if __name__ == '__main__':
    check_timestamps()