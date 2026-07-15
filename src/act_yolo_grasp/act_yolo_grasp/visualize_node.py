import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


class MultiCameraViewer(Node):
    def __init__(self):
        super().__init__("multi_camera_viewer")

        self.bridge = CvBridge()

        self.frames: dict[str, np.ndarray | None] = {
            "front": None,
            "top": None,
            "left_wrist": None,
            "right_wrist": None,
        }

        self.mask_frames: dict[str, np.ndarray | None] = {
            "front_mask": None,
            "top_mask": None,
            "left_wrist_mask": None,
            "right_wrist_mask": None,
        }

        # EMA coefficient: smaller = smoother, larger = more responsive
        self.ema_alpha = 0.2

        # FPS for display (after EMA)
        self.fps = {
            "front": 0.0,
            "top": 0.0,
            "left_wrist": 0.0,
            "right_wrist": 0.0,
            "front_mask": 0.0,
            "top_mask": 0.0,
            "left_wrist_mask": 0.0,
            "right_wrist_mask": 0.0,
        }

        # raw instantaneous FPS
        self.fps_raw = {
            "front": 0.0,
            "top": 0.0,
            "left_wrist": 0.0,
            "right_wrist": 0.0,
            "front_mask": 0.0,
            "top_mask": 0.0,
            "left_wrist_mask": 0.0,
            "right_wrist_mask": 0.0,
        }

        # last update time per view
        self.last_time: dict[str, float | None] = {
            "front": None,
            "top": None,
            "left_wrist": None,
            "right_wrist": None,
            "front_mask": None,
            "top_mask": None,
            "left_wrist_mask": None,
            "right_wrist_mask": None,
        }

        self.create_subscription(Image, "/front/image_raw", self.cb_front, 10)
        self.create_subscription(Image, "/front/YOLO_mask", self.cb_front_mask, 10)
        self.create_subscription(Image, "/top/top_realsense_node/color/image_raw", self.cb_top, 10)
        self.create_subscription(Image, "/top/YOLO_mask", self.cb_top_mask, 10)
        self.create_subscription(Image, "/left_wrist/left_wrist_realsense_node/color/image_raw", self.cb_left, 10)
        self.create_subscription(Image, "/left_wrist/YOLO_mask", self.cb_left_mask, 10)
        self.create_subscription(Image, "/right_wrist/right_wrist_realsense_node/color/image_raw", self.cb_right, 10)
        self.create_subscription(Image, "/right_wrist/YOLO_mask", self.cb_right_mask, 10)

        self.ensemble_std_window_mean = float("nan")
        self.confidence_threshold = 0.03  # same threshold as the ACT node

        self.create_subscription(
            Float32MultiArray,
            "/ensemble_std_window_mean",
            self.cb_ensemble_std_window_mean,
            10
        )

        self.timer = self.create_timer(0.03, self.show_images)

        # single window name
        self.window_name = "Multi Camera Viewer"

        # display size of each sub-view
        self.tile_w = 320
        self.tile_h = 240

    def update_fps(self, key):
        now = time.time()

        last = self.last_time[key]
        if last is not None:
            dt = now - last
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps_raw[key] = instant_fps

                # assign the first valid value directly instead of ramping up from 0
                if self.fps[key] == 0.0:
                    self.fps[key] = instant_fps
                else:
                    self.fps[key] = (
                        self.ema_alpha * instant_fps
                        + (1.0 - self.ema_alpha) * self.fps[key]
                    )

        self.last_time[key] = now

    @staticmethod
    def draw_info(frame: np.ndarray, cam_name, fps_value):
        img = frame.copy()

        # camera name
        cv2.putText(
            img,
            cam_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # FPS
        cv2.putText(
            img,
            f"FPS: {fps_value:.1f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return img

    def make_blank(self, text):
        img = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (30, self.tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )
        return img

    def prepare_tile(self, frame: np.ndarray | None, cam_name, fps_value):
        if frame is None:
            return self.make_blank(f"{cam_name}: No Image")

        show = np.asarray(cv2.resize(frame, (self.tile_w, self.tile_h)))
        show = self.draw_info(show, cam_name, fps_value)
        return show

    def cb_front(self, msg: Image):
        self.frames["front"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("front")

    def cb_front_mask(self, msg: Image):
        self.mask_frames["front_mask"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("front_mask")

    def cb_top(self, msg: Image):
        self.frames["top"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("top")

    def cb_top_mask(self, msg: Image):
        self.mask_frames["top_mask"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("top_mask")

    def cb_left(self, msg: Image):
        self.frames["left_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("left_wrist")

    def cb_left_mask(self, msg: Image):
        self.mask_frames["left_wrist_mask"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("left_wrist_mask")

    def cb_right(self, msg: Image):
        self.frames["right_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("right_wrist")

    def cb_right_mask(self, msg: Image):
        self.mask_frames["right_wrist_mask"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.update_fps("right_wrist_mask")

    def cb_ensemble_std_window_mean(self, msg: Float32MultiArray):
        if len(msg.data) > 0:
            self.ensemble_std_window_mean = float(msg.data[0])

    def show_images(self):
        # currently tiles the mask views
        front = self.prepare_tile(
            self.mask_frames["front_mask"],
            "Front",
            self.fps["front_mask"]
        )

        # overlay ensemble_std_window_mean at the top-right of the front view
        val = self.ensemble_std_window_mean
        if not np.isnan(val):
            color = (0, 0, 255) if val >= self.confidence_threshold else (0, 255, 0)
            text = f"win_mean: {val:.4f}"
        else:
            color = (200, 200, 200)
            text = "win_mean: --"

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = self.tile_w - text_size[0] - 10
        y = 30
        cv2.putText(front, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        top = self.prepare_tile(
            self.mask_frames["top_mask"],
            "Top",
            self.fps["top_mask"]
        )
        left = self.prepare_tile(
            self.mask_frames["left_wrist_mask"],
            "Left Wrist",
            self.fps["left_wrist_mask"]
        )
        right = self.prepare_tile(
            self.mask_frames["right_wrist_mask"],
            "Right Wrist",
            self.fps["right_wrist_mask"]
        )

        top_row = np.hstack((front, top))
        bottom_row = np.hstack((left, right))
        canvas = np.vstack((top_row, bottom_row))

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
