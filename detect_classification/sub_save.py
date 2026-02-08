from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO
# 훈련 데이터 생성
bridge = CvBridge()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_dir = Path.home() / "rgb_images"
save_dir.mkdir(parents=True, exist_ok=True)

model = YOLO('yolov8s.pt')

SAVE_COOLDOWN_SEC = 0.05

# 이미지 대비 박스 크기
MIN_BOX_AREA_RATIO = 0.03
MIN_BOX_W_RATIO = 0.12
MIN_BOX_H_RATIO = 0.12

class CameraListener(Node):
    def __init__(self):
        super().__init__('camera_listener')
        self.subscription = self.create_subscription(Image, '/rgb', self.listener_callback, 10)
        self.count = 0
        self.last_save_t = 0.0
        self.get_logger().info(f"Saving to: {str(save_dir)}")

    def listener_callback(self, msg: Image): 
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_image is None:
                return

            H, W = cv_image.shape[:2]
            img_area = float(W * H)

            results = model.predict(
                source=cv_image,
                conf=0.15,
                save=False,
                device=device,
                verbose=False
            )

            if not results or results[0].boxes is None:
                print("detected boxes: 0")
                return

            xyxy = results[0].boxes.xyxy
            n = int(xyxy.shape[0])

            print(f"\n[FRAME] detected boxes = {n}")

            large_box_found = False
            best_ratio = 0.0

            for i in range(n):
                x1, y1, x2, y2 = xyxy[i].tolist()
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)

                box_area = bw * bh
                area_ratio = box_area / img_area
                w_ratio = bw / W
                h_ratio = bh / H

                best_ratio = max(best_ratio, area_ratio)

                print(
                    f"  box[{i}] "
                    f"area_ratio={area_ratio:.4f} | "
                    f"w_ratio={w_ratio:.3f} | "
                    f"h_ratio={h_ratio:.3f}"
                )

                # 크기 조건
                if (area_ratio >= MIN_BOX_AREA_RATIO and
                    w_ratio >= MIN_BOX_W_RATIO and
                    h_ratio >= MIN_BOX_H_RATIO):
                    large_box_found = True

            print(f"  -> max area_ratio in frame = {best_ratio:.4f}")

            now = time.time()
            if large_box_found and (now - self.last_save_t) >= SAVE_COOLDOWN_SEC:
                file_path = save_dir / f"frame_{self.count:04d}.png"
                if cv2.imwrite(str(file_path), cv_image):
                    print(f"SAVED ({file_path.name})")
                    self.count += 1
                    self.last_save_t = now
                else:
                    print("SAVE FAILED")

        except Exception as e:
            print(f"[ERROR] {repr(e)}")

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

