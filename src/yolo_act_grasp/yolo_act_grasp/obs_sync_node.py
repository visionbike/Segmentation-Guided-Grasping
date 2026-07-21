import time
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ObservationSyncNode(Node):
    """
    Time-align the camera streams with the latest joint state into one observation.

    Subscribes to one color stream per configured camera and to the 17-dim qpos published by JointStateReaderNode.
    An ApproximateTimeSynchronizer groups image frames whose header stamps agree within ``image_slop``; when such
    a group arrives, the cached qpos is checked for freshness (``encoder_timeout``) and, if valid, the images and
    the qpos are republished together under ``/sync/``.

    Any observation that fails either check is dropped whole, so everything downstream (yolo_seg, act_policy) sees
    a consistent snapshot: these images and this joint state belong to the same moment. Without joint-state feedback
    the node publishes nothing at all -- the 1 Hz ``[SYNC]`` stats line reports the drop counters so that silence is
    diagnosable.
    """
    def __init__(self):
        super().__init__("obs_sync_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("encoder_timeout", 0.2)      # max qpos age (s) before dropping
        self.declare_parameter("image_slop", 0.1)           # max stamp spread (s) between cameras
        self.declare_parameter("queue_size", 20)            # synchronizer history per camera

        # list of cameras to synchronize; add as many as connected
        # e.g. ["top", "left_wrist", "right_wrist"] or add "front"
        # self.declare_parameter("camera_names", ["top", "left_wrist", "right_wrist", "front"])
        self.declare_parameter("camera_names", ["top", "left_wrist", "right_wrist"])

        # input topic per camera
        self.declare_parameter("camera_top_topic",   "/cameras/top/color/image_raw")
        self.declare_parameter("camera_left_topic",  "/cameras/left_wrist/color/image_raw")
        self.declare_parameter("camera_right_topic", "/cameras/right_wrist/color/image_raw")
        self.declare_parameter("camera_front_topic", "/cameras/front/color/image_raw")
        self.declare_parameter("encoder_topic", "/joint_state_feedback")

        # output (sync) topic per camera
        self.declare_parameter("sync_top_topic",   "/sync/top/image_raw")
        self.declare_parameter("sync_left_topic",  "/sync/left_wrist/image_raw")
        self.declare_parameter("sync_right_topic", "/sync/right_wrist/image_raw")
        self.declare_parameter("sync_front_topic", "/sync/front/image_raw")
        self.declare_parameter("sync_qpos_topic",  "/sync/qpos")

        self.declare_parameter("print_timing_log", True)

        self.encoder_timeout  = self.get_parameter("encoder_timeout").get_parameter_value().double_value
        self.image_slop       = self.get_parameter("image_slop").get_parameter_value().double_value
        self.queue_size       = self.get_parameter("queue_size").get_parameter_value().integer_value
        self.print_timing_log = self.get_parameter("print_timing_log").get_parameter_value().bool_value
        self.encoder_topic    = self.get_parameter("encoder_topic").get_parameter_value().string_value
        self.sync_qpos_topic  = self.get_parameter("sync_qpos_topic").get_parameter_value().string_value

        # read the camera list (STRING_ARRAY)
        self.active_cameras = list(self.get_parameter("camera_names").get_parameter_value().string_array_value)
        if not self.active_cameras:
            raise RuntimeError("[NODE] camera_names is empty; expected a list like ['top', 'left_wrist']")

        # name -> input topic
        self.camera_input_topics = {
            "top":         self.get_parameter("camera_top_topic").get_parameter_value().string_value,
            "left_wrist":  self.get_parameter("camera_left_topic").get_parameter_value().string_value,
            "right_wrist": self.get_parameter("camera_right_topic").get_parameter_value().string_value,
            "front":       self.get_parameter("camera_front_topic").get_parameter_value().string_value
        }

        # name -> output topic
        self.camera_sync_topics = {
            "top":         self.get_parameter("sync_top_topic").get_parameter_value().string_value,
            "left_wrist":  self.get_parameter("sync_left_topic").get_parameter_value().string_value,
            "right_wrist": self.get_parameter("sync_right_topic").get_parameter_value().string_value,
            "front":       self.get_parameter("sync_front_topic").get_parameter_value().string_value,
        }

        unknown = [cam for cam in self.active_cameras if cam not in self.camera_sync_topics]
        if unknown:
            raise RuntimeError(f"[NODE] unknown cameras ({unknown}); known: {sorted(self.camera_sync_topics)}")

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------
        self.latest_encoder_msg: Float32MultiArray | None = None
        self.latest_encoder_recv_time: float = 0.0

        # diagnostics counters
        self.publish_count: int = 0
        self.drop_no_encoder: int = 0
        self.drop_old_encoder: int = 0
        self.last_stat_time: float = time.time()

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
        # Diagnostics timer (wall-clock, so it reports even when nothing syncs)
        # --------------------------------------------------
        self.stats_timer = self.create_timer(1.0, self.log_stats_callback)

        self.get_logger().info("[NODE] =========================================")
        self.get_logger().info("[NODE] ObservationSyncNode ROS2 node started")
        self.get_logger().info(f"[NODE] active cameras ({len(self.active_cameras)}): {self.active_cameras}")
        for cam in self.active_cameras:
            self.get_logger().info(
                f"[NODE]   {cam}: {self.camera_input_topics[cam]}"
                f" -> {self.camera_sync_topics[cam]}"
            )
        self.get_logger().info(f"[NODE] encoder_topic   : {self.encoder_topic}")
        self.get_logger().info(f"[NODE] sync_qpos_topic : {self.sync_qpos_topic}")
        self.get_logger().info(f"[NODE] image_slop      : {self.image_slop}")
        self.get_logger().info(f"[NODE] encoder_timeout : {self.encoder_timeout}")
        self.get_logger().info(f"[NODE] queue_size      : {self.queue_size}")
        self.get_logger().info("[NODE] =========================================")

    def encoder_callback(self, msg: Float32MultiArray) -> None:
        """
        Cache the latest joint state and its arrival time.

        The qpos message carries no header stamp, so it cannot take part in the image
        synchronizer; freshness is judged from wall-clock arrival instead.

        :param msg: 17-dim joint state published by JointStateReader.
        """
        self.latest_encoder_msg = msg
        self.latest_encoder_recv_time = time.time()

    def log_stats_callback(self) -> None:
        """
        Report publish/drop counters once per second, then reset them.

        Runs on a wall-clock timer rather than inside the sync callback, so the counters are
        still reported when every observation is being dropped (or when no images arrive at
        all) -- exactly the cases where the pipeline looks silent.
        """
        self.get_logger().info(
            f"[SYNC] publish={self.publish_count} Hz, "
            f"drop_no_encoder={self.drop_no_encoder}, "
            f"drop_old_encoder={self.drop_old_encoder}"
        )
        self.publish_count = 0
        self.drop_no_encoder = 0
        self.drop_old_encoder = 0
        self.last_stat_time = time.time()

    def image_sync_callback(self, *args: Image) -> None:
        """
        Publish one matched image set together with the cached qpos.

        Called by the synchronizer with one Image per active camera, in ``active_cameras`` order.
        The whole observation is dropped if no joint state has arrived yet, or if the cached one
        is older than ``encoder_timeout``.

        :param args: one Image per configured camera, ordered as ``active_cameras``.
        """
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
                    f"[SYNC] image_spread={image_spread:.4f}s > slop={self.image_slop:.4f}s",
                    throttle_duration_sec=1.0
                )


def main(args=None):
    logger = get_logger("obs_sync_node")
    rclpy.init(args=args)
    node = None

    try:
        node = ObservationSyncNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("[NODE] Ctrl-C received, shutting down.")
    except Exception as e:
        logger.fatal(f"[NODE] Node crashed: {e}.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()  # guard avoids double-shutdown error


if __name__ == "__main__":
    main()
