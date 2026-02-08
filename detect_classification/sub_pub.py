from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import time
import cv2
import numpy as np

from ultralytics import YOLO
from torchvision import models, transforms

# ADD DB
import sqlite3
from pathlib import Path

bridge = CvBridge()
device = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_PATH = "/home/rokey/Desktop/detect_classification/yolov8s.pt"
CLASSIFIER_PT   = "/home/rokey/Desktop/detect_classification/best_resnet50_abc.pt"
CLASS_NAMES     = ["A", "B", "C"]

# YOLO
model = YOLO(YOLO_MODEL_PATH)
PUBLISH_COOLDOWN_SEC = 3.5

# 최소 박스 크기
MIN_BOX_AREA_RATIO = 0.03
MIN_BOX_W_RATIO = 0.12
MIN_BOX_H_RATIO = 0.12

# crop
CROP_PAD_RATIO = 0.10

# 분류 confidence 너무 낮으면 publish 안 함
MIN_CLS_CONF = 0.0

# ✅ [ADD] DB 경로 (원하면 절대경로로 바꿔도 됨)
DB_PATH = str(Path(__file__).with_name("apple_grade_counts.db"))


def build_resnet50(num_classes: int):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(pt_path: str, num_classes: int):
    ckpt = torch.load(pt_path, map_location="cpu")
    m = build_resnet50(num_classes)

    if isinstance(ckpt, dict):
        if "fc.weight" in ckpt or "layer1.0.conv1.weight" in ckpt:
            sd = ckpt
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            sd = ckpt["model_state"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            raise RuntimeError(f"Unknown checkpoint dict keys: {list(ckpt.keys())[:20]}")

        # DataParallel prefix 제거
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing:
            print("[WARN] missing keys:", missing[:20])
        if unexpected:
            print("[WARN] unexpected keys:", unexpected[:20])

    else:
        if hasattr(ckpt, "state_dict"):
            m = ckpt
        else:
            raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    m.eval()
    m.to(device)
    return m


classifier = load_classifier(CLASSIFIER_PT, num_classes=len(CLASS_NAMES))

cls_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.inference_mode()
def classify_bgr_crop(crop_bgr: np.ndarray):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = cls_tf(rgb).unsqueeze(0).to(device)
    logits = classifier(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    label = CLASS_NAMES[int(idx)]
    return label, float(conf.item())


def crop_with_pad(img_bgr, x1, y1, x2, y2, pad_ratio=0.1):
    H, W = img_bgr.shape[:2]
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    nx1 = max(0, int(x1) - pad_x)
    ny1 = max(0, int(y1) - pad_y)
    nx2 = min(W - 1, int(x2) + pad_x)
    ny2 = min(H - 1, int(y2) + pad_y)

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return img_bgr[ny1:ny2, nx1:nx2]


class CameraListener(Node):
    def __init__(self):
        super().__init__("camera_listener")

        # 입력 이미지 구독
        self.subscription = self.create_subscription(Image, "/rgb", self.listener_callback, 10)

        # 등급 결과 퍼블리셔
        self.pub_grade = self.create_publisher(String, "/apple/grade", 10)

        self.last_pub_t = 0.0
        self.pub_count = 0

        # DB 초기화
        self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._db_init()

        self.get_logger().info("Subscribing: /rgb")
        self.get_logger().info("Publishing: /apple/grade")
        self.get_logger().info(f"DB: {DB_PATH}")

    # DB 테이블/초기값 생성
    def _db_init(self):
        cur = self.db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS grade_counts (
                grade TEXT PRIMARY KEY,
                cnt   INTEGER NOT NULL
            )
        """)
        # A/B/C 없으면 0으로 삽입
        for g in CLASS_NAMES:
            cur.execute(
                "INSERT OR IGNORE INTO grade_counts(grade, cnt) VALUES(?, ?)",
                (g, 0),
            )
        self.db.commit()

    # 카운트 +1 하고 현재 값 반환
    def _db_inc_and_get(self, grade: str) -> int:
        cur = self.db.cursor()
        cur.execute("UPDATE grade_counts SET cnt = cnt + 1 WHERE grade = ?", (grade,))
        # 혹시 grade row가 없다면(이론상 없어야 함) 보정
        if cur.rowcount == 0:
            cur.execute("INSERT INTO grade_counts(grade, cnt) VALUES(?, ?)", (grade, 1))
        self.db.commit()
        cur.execute("SELECT cnt FROM grade_counts WHERE grade = ?", (grade,))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    # 전체 카운트 읽기
    def _db_get_all(self) -> dict:
        cur = self.db.cursor()
        cur.execute("SELECT grade, cnt FROM grade_counts")
        return {g: int(c) for (g, c) in cur.fetchall()}

    def listener_callback(self, msg: Image):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if cv_image is None:
                return

            H, W = cv_image.shape[:2]
            img_area = float(W * H)

            results = model.predict(
                source=cv_image,
                conf=0.15,
                save=False,
                device=device,
                verbose=False,
            )

            if not results or results[0].boxes is None:
                self.get_logger().debug("detected boxes: 0")
                return

            xyxy = results[0].boxes.xyxy
            n = int(xyxy.shape[0])

            large_box_found = False
            best_area = 0.0
            best_box = None
            best_i = -1

            # 조건 만족한 것 중 가장 큰 박스 고르기
            for i in range(n):
                x1, y1, x2, y2 = xyxy[i].tolist()
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)

                box_area = bw * bh
                area_ratio = box_area / img_area
                w_ratio = bw / W
                h_ratio = bh / H

                if (
                    area_ratio >= MIN_BOX_AREA_RATIO
                    and w_ratio >= MIN_BOX_W_RATIO
                    and h_ratio >= MIN_BOX_H_RATIO
                ):
                    large_box_found = True
                    if box_area > best_area:
                        best_area = box_area
                        best_box = (x1, y1, x2, y2)
                        best_i = i

            grade_label, grade_conf = None, None
            if large_box_found and best_box is not None:
                x1, y1, x2, y2 = best_box
                crop = crop_with_pad(cv_image, x1, y1, x2, y2, pad_ratio=CROP_PAD_RATIO)
                if crop is not None:
                    grade_label, grade_conf = classify_bgr_crop(crop)
                else:
                    self.get_logger().warn("crop invalid → classification skipped")

            now = time.time()
            if large_box_found and (now - self.last_pub_t) >= PUBLISH_COOLDOWN_SEC:
                if grade_label is None:
                    self.get_logger().warn("publish 조건 만족했지만 grade_label=None → publish skip")
                    return

                if grade_conf is not None and grade_conf < MIN_CLS_CONF:
                    self.get_logger().info(
                        f"cls conf too low ({grade_conf:.3f} < {MIN_CLS_CONF:.3f}) → publish skip"
                    )
                    return

                # 등급 퍼블리시
                s = String()
                s.data = grade_label
                self.pub_grade.publish(s)

                self.pub_count += 1
                self.last_pub_t = now

                # 퍼블리시 성공 시점에 DB 카운트 +1
                new_cnt = self._db_inc_and_get(grade_label)
                all_cnt = self._db_get_all()

                # 퍼블리시할 때 get_logger로 등급 출력 + DB 카운트 출력
                self.get_logger().info(
                    f"PUBLISH /apple/grade: {grade_label} (conf={grade_conf:.3f}) | "
                    f"box={best_i} | pub_count={self.pub_count} | "
                    f"DB {grade_label}={new_cnt} | ALL={all_cnt}"
                )

        except Exception as e:
            self.get_logger().error(f"[ERROR] {repr(e)}")

    # 종료 시 DB 닫기
    def destroy_node(self):
        try:
            if hasattr(self, "db") and self.db is not None:
                self.db.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

