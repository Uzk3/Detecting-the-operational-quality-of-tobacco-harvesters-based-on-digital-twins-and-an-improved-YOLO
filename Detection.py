import socket
import threading
import cv2
import math
import time
from ultralytics import YOLO


UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock_detection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

GPS_UDP_PORT = 50001
sock_gps = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_gps.bind(('0.0.0.0', GPS_UDP_PORT))


MODEL_PATH = "C:/Users/Uzk3/runs/detect/3shengbao/best.pt"


STREAM_URLS = [
    "http://192.168.137.8:8082/?action=stream",
    "http://192.168.137.8:8080/?action=stream",
    "http://192.168.137.8:8081/?action=stream"
]


CONF_THRESHOLD = 0.5
ROI_RATIO = 0.6
TRACKER_CFG = "bytetrack.yaml"


gps_lock = threading.Lock()
latitude, longitude = None, None
prev_lat, prev_lon = None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def nmea_to_decimal(degree_minute, direction):
    degree_minute = float(degree_minute)
    degree = int(degree_minute // 100)
    minute = degree_minute - degree * 100
    value = degree + minute / 60
    if direction in ['S', 'W']:
        value = -value
    return value

def parse_gnrmc(line):
    if not line.startswith('$GNRMC'):
        return None
    parts = line.split(',')
    if len(parts) < 7 or parts[2] != 'A':
        return None
    lat = nmea_to_decimal(parts[3], parts[4])
    lon = nmea_to_decimal(parts[5], parts[6])
    return lat, lon


def gps_receiver_loop():
    global latitude, longitude, prev_lat, prev_lon
    DIST_THRESHOLD = 0.1  
    while True:
        data, addr = sock_gps.recvfrom(1024)
        line = data.decode(errors='ignore').strip()
        result = parse_gnrmc(line)
        if result:
            with gps_lock:
                lat, lon = result
                if latitude is not None and longitude is not None:
                    dist = haversine(latitude, longitude, lat, lon)
                    # print(f"移动距离: {dist:.3f}m")
                    if dist >= DIST_THRESHOLD:
                        print(f"🌍 : {latitude},{longitude} → {lat},{lon},  {dist:.3f}m，发送direction1")
                        sock_detection.sendto(b"direction1", (UDP_IP, UDP_PORT))
                latitude, longitude = lat, lon


seen_ids = [{}, {}, {}]


def send_udp(msg):
    print(f"🔔 UDP: {msg}")
    sock_detection.sendto(msg.encode(), (UDP_IP, UDP_PORT))


TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS


def process_stream(stream_id, url):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ can not open video: {url}")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ ")
        return

    height, width = frame.shape[:2]
    window_name = f"Stream-{stream_id+1}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_frame_time = time.time()

    while True:
        current_time = time.time()
        cap.grab()

        if current_time - last_frame_time < FRAME_INTERVAL:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        results = model.track(
            source=frame,
            tracker=TRACKER_CFG,
            persist=True,
            conf=CONF_THRESHOLD,
            stream=False
        )[0]

        boxes = results.boxes
        current_seen = seen_ids[stream_id]

        if stream_id == 0:
            mid_x = width // 2
            cv2.line(frame, (mid_x, 0), (mid_x, height), (255, 0, 0), 2)
        elif stream_id in [1, 2]:
            roi_w = int(width * ROI_RATIO)
            start_x = (width - roi_w) // 2
            end_x = start_x + roi_w
            cv2.rectangle(frame, (start_x, 0), (end_x, height), (255, 0, 0), 2)

        for box in boxes:
            if not hasattr(box, 'id') or box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls.cpu().numpy())
            trk_id = int(box.id.cpu().numpy())

            color = (0, 255, 0) if cls_id in [0, 2] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{trk_id} C:{cls_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if trk_id in current_seen:
                continue

            if stream_id == 0:
                position = "left" if (x1 + x2) // 2 < mid_x else "right"
                if cls_id == 2:
                    send_udp("drop_left_" if position == "left" else "drop_right")
                elif cls_id == 3:
                    send_udp("dmg_left__" if position == "left" else "dmg_right_")
            elif stream_id == 1 and cls_id in [0, 1]:
                if start_x <= x1 <= end_x:
                    send_udp("leak_left_" if cls_id == 0 else "normalleft")
            elif stream_id == 2 and cls_id in [0, 1]:
                if start_x <= x1 <= end_x:
                    send_udp("leak_right" if cls_id == 0 else "normaright")

            current_seen[trk_id] = True

        cv2.imshow(window_name, frame)
        last_frame_time = current_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=gps_receiver_loop, daemon=True).start()
    for i, url in enumerate(STREAM_URLS):
        threading.Thread(target=process_stream, args=(i, url), daemon=True).start()
    while True:
        time.sleep(1)
