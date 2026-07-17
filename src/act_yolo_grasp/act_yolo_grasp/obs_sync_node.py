import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RawObservationSyncNode(Node):
    def __init__(self):
        super().__init__("raw_observation_sync_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("encoder_timeout", 0.2)
        self.declare_parameter("image_slop", 0.2)
        self.declare_parameter("queue_size", 20)

        # list of cameras to synchronize; add as many as connected
        # e.g. ["top", "left_wrist", "right_wrist"] or add "front"
        # self.declare_parameter("camera_names", ["top", "left_wrist", "right_wrist", "front"])
        self.declare_parameter("camera_names", ["top", "left_wrist", "right_wrist"])

        # input topic per camera
        self.declare_parameter("top_topic",   "/top/top_realsense_node/color/image_raw")
        self.declare_parameter("left_topic",  "/left_wrist/left_wrist_realsense_node/color/image_raw")
        self.declare_parameter("right_topic", "/right_wrist/right_wrist_realsense_node/color/image_raw")
        self.declare_parameter("front_topic", "/front/image_raw")
        self.declare_parameter("encoder_topic", "/motor_angle_feedback_topic")

        # output (sync) topic per camera
        self.declare_parameter("sync_top_topic",   "/sync/top/image_raw")
        self.declare_parameter("sync_left_topic",  "/sync/left_wrist/image_raw")
        self.declare_parameter("sync_right_topic", "/sync/right_wrist/image_raw")
        self.declare_parameter("sync_front_topic", "/sync/front/image_raw")
        self.declare_parameter("sync_qpos_topic",  "/sync/qpos")

        self.declare_parameter("print_timing_log", True)

        self.encoder_timeout  = float(self.get_parameter("encoder_timeout").value)
        self.image_slop       = float(self.get_parameter("image_slop").value)
        self.queue_size       = int(self.get_parameter("queue_size").value)
        self.print_timing_log = bool(self.get_parameter("print_timing_log").value)
        self.encoder_topic    = str(self.get_parameter("encoder_topic").value)
        self.sync_qpos_topic  = str(self.get_parameter("sync_qpos_topic").value)

        # read the camera list (STRING_ARRAY)
        self.active_cameras = list(self.get_parameter("camera_names").value)

        # name -> input topic
        self.camera_input_topics = {
            "top":         str(self.get_parameter("top_topic").value),
            "left_wrist":  str(self.get_parameter("left_topic").value),
            "right_wrist": str(self.get_parameter("right_topic").value),
            "front":       str(self.get_parameter("front_topic").value),
        }

        # name -> output topic
        self.camera_sync_topics = {
            "top":         str(self.get_parameter("sync_top_topic").value),
            "left_wrist":  str(self.get_parameter("sync_left_topic").value),
            "right_wrist": str(self.get_parameter("sync_right_topic").value),
            # "front":       str(self.get_parameter("sync_front_topic").value),
        }

        # --------------------------------------------------
        # Encoder cache
        # --------------------------------------------------
        self.latest_encoder_msg: Float32MultiArray | None = None
        self.latest_encoder_recv_time = 0.0

        # --------------------------------------------------
        # Publishers (created only for active cameras)
        # --------------------------------------------------
        self.cam_pubs = {
            cam: self.create_publisher(Image, self.camera_sync_topics[cam], 10)
            for cam in self.active_cameras
        }
        self.qpos_pub = self.create_publisher(Float32MultiArray, self.sync_qpos_topic, 10)

        # --------------------------------------------------
        # Encoder subscriber
        # --------------------------------------------------
        self.encoder_sub = self.create_subscription(
            Float32MultiArray,
            self.encoder_topic,
            self.encoder_callback,
            10,
        )

        # --------------------------------------------------
        # Image subscribers + synchronizer (dynamic: as many as configured)
        # --------------------------------------------------
        self.cam_subs = [
            Subscriber(self, Image, self.camera_input_topics[cam])
            for cam in self.active_cameras
        ]

        self.sync = ApproximateTimeSynchronizer(
            self.cam_subs,
            queue_size=self.queue_size,
            slop=self.image_slop,
        )
        # message_filters' stubs can't express a variadic callback for a
        # runtime-configured number of cameras
        # noinspection PyTypeChecker
        self.sync.registerCallback(self.image_sync_callback)

        # --------------------------------------------------
        # Debug counters
        # --------------------------------------------------
        self.publish_count    = 0
        self.drop_no_encoder  = 0
        self.drop_old_encoder = 0
        self.last_stat_time   = time.time()

        self.get_logger().info("=========================================")
        self.get_logger().info("Raw Observation Sync Node started")
        self.get_logger().info(f"active cameras ({len(self.active_cameras)}): {self.active_cameras}")
        for cam in self.active_cameras:
            self.get_logger().info(
                f"  {cam}: {self.camera_input_topics[cam]}"
                f" -> {self.camera_sync_topics[cam]}"
            )
        self.get_logger().info(f"encoder_topic   : {self.encoder_topic}")
        self.get_logger().info(f"sync_qpos_topic : {self.sync_qpos_topic}")
        self.get_logger().info(f"image_slop      : {self.image_slop}")
        self.get_logger().info(f"encoder_timeout : {self.encoder_timeout}")
        self.get_logger().info("=========================================")

    # --------------------------------------------------
    # Encoder callback
    # --------------------------------------------------
    def encoder_callback(self, msg: Float32MultiArray):
        self.latest_encoder_msg = msg
        self.latest_encoder_recv_time = time.time()

    # --------------------------------------------------
    # N-camera sync callback (one arg per configured camera)
    # --------------------------------------------------
    def image_sync_callback(self, *args: Image) -> None:
        now = time.time()

        encoder_msg = self.latest_encoder_msg
        if encoder_msg is None:
            self.drop_no_encoder += 1
            return

        encoder_age = now - self.latest_encoder_recv_time
        if encoder_age > self.encoder_timeout:
            self.drop_old_encoder += 1
            return

        for cam, msg in zip(self.active_cameras, args):
            self.cam_pubs[cam].publish(msg)

        self.qpos_pub.publish(encoder_msg)

        self.publish_count += 1

        if self.print_timing_log:
            stamps = [
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                for msg in args
            ]
            image_spread = max(stamps) - min(stamps)
            if image_spread > self.image_slop:
                self.get_logger().warning(
                    f"[SYNC] image_spread={image_spread:.4f}s > slop={self.image_slop:.4f}s"
                )

        if now - self.last_stat_time >= 1.0:
            # self.get_logger().info(
            #     f"[SYNC RATE] publish={self.publish_count} Hz, "
            #     f"drop_no_encoder={self.drop_no_encoder}, "
            #     f"drop_old_encoder={self.drop_old_encoder}"
            # )
            self.publish_count    = 0
            self.drop_no_encoder  = 0
            self.drop_old_encoder = 0
            self.last_stat_time   = now


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RawObservationSyncNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
