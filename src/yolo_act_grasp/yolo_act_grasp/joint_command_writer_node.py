import time
import math
import threading
import serial

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Bool

from .serial_protocol import encode_joint_command, clamp_joint_angles


class JointCommandWriterNode(Node):
    """
    Encode ACT joint commands and write them to the Teensy over serial.

    Outbound half of the U2D2 serial link and mirror of JointStateReader. Subscribes to
    ``joint_command_topic`` (17-dim action from act_policy), debounces the two gripper channels into
    stable 0/1 states, clamps every joint to its mechanical limit, encodes a 37-byte command packet
    (``serial_protocol.encode_joint_command``), and writes it every ``send_interval`` seconds.

    Also manages the episode handshake: publishes ``/episode_start`` on bring-up, waits
    ``startup_delay_sec`` before sending, then publishes ``/episode_record_begin`` once; a True on
    ``stop_signal_topic`` halts sending and shuts the node down.
    """
    def __init__(self):
        super().__init__("joint_command_writer_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("com_port", "/dev/teensy_serial")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("send_interval", 0.1)    # how often a command is written (unit: s)
        self.declare_parameter("serial_enable", True)
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("stop_signal_topic", "/episode_stop")
        self.declare_parameter("start_signal_topic", "/episode_start")
        self.declare_parameter("record_begin_topic", "/episode_record_begin")
        self.declare_parameter("startup_delay_sec", 3.0)

        # per-hand (left/right) close threshold
        self.declare_parameter("right_gripper_threshold", 0.5)
        self.declare_parameter("left_gripper_threshold", 0.5)

        # consecutive frames required before the gripper state switches
        self.declare_parameter("right_gripper_required_count", 1)
        self.declare_parameter("left_gripper_required_count", 1)

        self.com_port = self.get_parameter("com_port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.send_interval = self.get_parameter("send_interval").get_parameter_value().double_value
        self.serial_enable = self.get_parameter("serial_enable").get_parameter_value().bool_value
        self.joint_command_topic = self.get_parameter("joint_command_topic").get_parameter_value().string_value
        self.stop_signal_topic = self.get_parameter("stop_signal_topic").get_parameter_value().string_value
        self.start_signal_topic = self.get_parameter("start_signal_topic").get_parameter_value().string_value
        self.record_begin_topic = self.get_parameter("record_begin_topic").get_parameter_value().string_value
        self.startup_delay_sec = self.get_parameter("startup_delay_sec").get_parameter_value().double_value

        right_count = max(1, self.get_parameter("right_gripper_required_count").get_parameter_value().integer_value)
        left_count = max(1, self.get_parameter("left_gripper_required_count").get_parameter_value().integer_value)

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------
        self.serial_lock = threading.Lock()
        self.serial_conn: serial.Serial | None = None

        # latest command cached from the action topic (radians for arms/neck)
        self.right_arm_command: list | None = None
        self.left_arm_command: list | None = None
        self.right_gripper_command: float = 0.0
        self.left_gripper_command: float = 0.0
        self.neck_command: list | None = None

        self.stop_requested: bool = False

        # per-side gripper debounce (the 0/1 actually sent out)
        self.gripper_state = {"right": 0, "left": 0}
        self.gripper_high_count = {"right": 0, "left": 0}
        self.gripper_low_count = {"right": 0, "left": 0}
        self.gripper_threshold = {
            "right": self.get_parameter("right_gripper_threshold").get_parameter_value().double_value,
            "left": self.get_parameter("left_gripper_threshold").get_parameter_value().double_value,
        }
        self.gripper_required_count = {"right": right_count, "left": left_count}

        # mechanical joint limits (degrees)
        self.right_arm_limits = [
            (-90.0, 90.0),      # R1
            (-90.0, 5.0),       # R2
            (-90.0, 90.0),      # R3
            (-0.0, 150.0),      # R4
            (-90.0, 90.0),      # R5
            (-15.0, 15.0),      # R6
            (-30.0, 30.0),      # R7
        ]
        self.left_arm_limits = [
            (-90.0, 90.0),      # L1
            (-5.0, 90.0),       # L2
            (-90.0, 90.0),      # L3
            (-150.0, 0.0),      # L4
            (-90.0, 90.0),      # L5
            (-15.0, 15.0),      # L6
            (-30.0, 30.0),      # L7
        ]
        self.neck_limit = [(-90.0, 90.0)]

        self.startup_time: float = time.time()
        self.send_enable_after_time: float = self.startup_time + self.startup_delay_sec
        self.last_countdown_print_sec: int | None = None
        self.record_begin_sent: bool = False

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.command_sub = self.create_subscription(
            Float32MultiArray,
            self.joint_command_topic,
            self.command_callback,
            10
        )
        self.stop_signal_sub = self.create_subscription(
            Bool,
            self.stop_signal_topic,
            self.stop_signal_callback,
            10
        )

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.start_pub = self.create_publisher(Bool, self.start_signal_topic, latched_qos)
        self.record_begin_pub = self.create_publisher(Bool, self.record_begin_topic, 10)

        # --------------------------------------------------
        # Serial initialization + start handshake
        # --------------------------------------------------
        self.init_serial()
        self.publish_start_signal_once()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.timer = self.create_timer(self.send_interval, self.send_command_callback)

        self.get_logger().info("[NODE] =========================================")
        self.get_logger().info("[NODE] JointCommandWriter ROS2 node started")
        self.get_logger().info(f"[NODE] com_port                     : {self.com_port}")
        self.get_logger().info(f"[NODE] baudrate                     : {self.baudrate}")
        self.get_logger().info(f"[NODE] send_interval                : {self.send_interval}")
        self.get_logger().info(f"[NODE] serial_enable                : {self.serial_enable}")
        self.get_logger().info(f"[NODE] joint_command_topic          : {self.joint_command_topic}")
        self.get_logger().info(f"[NODE] stop_signal_topic            : {self.stop_signal_topic}")
        self.get_logger().info(f"[NODE] start_signal_topic           : {self.start_signal_topic}")
        self.get_logger().info(f"[NODE] record_begin_topic           : {self.record_begin_topic}")
        self.get_logger().info(f"[NODE] right_gripper_threshold      : {self.gripper_threshold['right']}")
        self.get_logger().info(f"[NODE] left_gripper_threshold       : {self.gripper_threshold['left']}")
        self.get_logger().info(f"[NODE] right_gripper_required_count : {self.gripper_required_count['right']}")
        self.get_logger().info(f"[NODE] left_gripper_required_count  : {self.gripper_required_count['left']}")
        self.get_logger().info("[NODE] =========================================")

    def init_serial(self) -> None:
        """
        Open the serial port to the Teensy (write side).
        """
        if self.serial_enable:
            try:
                self.serial_conn = serial.Serial(
                    self.com_port,
                    self.baudrate,
                    timeout=1,
                    write_timeout=1
                )
                self.get_logger().info("[SERIAL] Serial enabled.")
            except Exception as e:
                self.serial_conn = None
                self.get_logger().error(f"[SERIAL] Failed to open port {self.com_port}: {e}.")
        else:
            self.serial_conn = None
            self.get_logger().warning("[SERIAL] Serial disabled (serial_enable=False).")

    def publish_start_signal_once(self) -> None:
        """
        Publish the episode start signal a few times so late subscribers catch it.
        """
        msg = Bool()
        msg.data = True
        for _ in range(5):
            self.start_pub.publish(msg)
            time.sleep(0.05)

        self.get_logger().info(
            f"[STARTUP] Published {self.start_signal_topic} = True, "
            f"sending starts after {self.startup_delay_sec:.1f} s."
        )

    def stop_signal_callback(self, msg: Bool) -> None:
        """
        On a True stop signal, cancel the timer, close serial, and shut the node down.

        :param msg: Bool stop signal; only True triggers a shutdown.
        """
        if not msg.data or self.stop_requested:
            return

        self.stop_requested = True
        self.get_logger().warning("[STOP] Received stop signal, halting sender.")

        try:
            self.timer.cancel()
        except Exception as e:
            self.get_logger().warning(f"[STOP] Failed to cancel timer {e}.")

        try:
            if self.serial_conn is not None and self.serial_conn.is_open:
                self.serial_conn.close()
                self.get_logger().info("[SERIAL] Serial port closed by stop signal.")
        except Exception as e:
            self.get_logger().warning(f"[SERIAL] Failed to close serial port on stop: {e}.")

        if rclpy.ok():
            rclpy.shutdown()

    def command_callback(self, msg: Float32MultiArray) -> None:
        """
        Cache the latest 17-dim joint command and update the gripper debounce.

        NOTE: the action vector order is ``[L1-L7, R1-R7, L_grip, R_grip, Neck]`` (left arm first).
        This must match act_policy's output ordering, which differs from the qpos feedback order
        published by JointStateReader. Verify before changing.

        :param msg: Float32MultiArray with 17 elements (radians for arms/neck, 0..1 for grippers).
        """
        if self.stop_requested:
            return

        data = list(msg.data)
        if len(data) < 17:
            return

        self.left_arm_command = data[0: 7]
        self.right_arm_command = data[7: 14]
        self.left_gripper_command = float(data[14])
        self.right_gripper_command = float(data[15])
        self.neck_command = [data[16]]

        # update gripper state on every new message
        self.update_gripper_state("right", self.right_gripper_command)
        self.update_gripper_state("left", self.left_gripper_command)

    def update_gripper_state(self, side: str, value: float) -> None:
        """
        Debounce one gripper channel: flip its 0/1 state only after
        ``gripper_required_count[side]`` consecutive frames past the threshold.

        :param side: "right" or "left".
        :param value: raw gripper prediction (0..1).
        """
        if value >= self.gripper_threshold[side]:
            self.gripper_high_count[side] += 1
            self.gripper_low_count[side] = 0
        else:
            self.gripper_low_count[side] += 1
            self.gripper_high_count[side] = 0

        if self.gripper_high_count[side] >= self.gripper_required_count[side]:
            if self.gripper_state[side] != 1:
                self.get_logger().info(
                    f"[GRIPPER] {side} state 0 -> 1 | value={value:.3f}, "
                    f"high_count={self.gripper_high_count[side]}."
                )
            self.gripper_state[side] = 1
        elif self.gripper_low_count[side] >= self.gripper_required_count[side]:
            if self.gripper_state[side] != 0:
                self.get_logger().info(
                    f"[GRIPPER] {side} state 1 -> 0 | value={value:.3f}, "
                    f"low_count={self.gripper_low_count[side]}."
                )
            self.gripper_state[side] = 0

    def send_command_callback(self) -> None:
        """
        Timer: honor the startup delay, emit the record-begin signal once, then send the
        latest cached command.
        """
        if self.stop_requested:
            return

        now = time.time()
        if now < self.send_enable_after_time:
            remain = int(math.ceil(self.send_enable_after_time - now))
            if self.last_countdown_print_sec != remain:
                self.get_logger().info(f"[STARTUP] Sending starts in {remain} second(s)...")
                self.last_countdown_print_sec = remain
            return

        if not self.record_begin_sent:
            msg = Bool()
            msg.data = True
            for _ in range(5):
                self.record_begin_pub.publish(msg)
                time.sleep(0.02)
            self.record_begin_sent = True
            self.get_logger().info(f"[RECORD] Published {self.record_begin_topic} = True.")

        if self.serial_conn is None or not self.serial_conn.is_open:
            return

        if self.right_arm_command is None or self.left_arm_command is None or self.neck_command is None:
            self.get_logger().warning("[PACKET] No command data yet.", throttle_duration_sec=1.0)
            return

        try:
            self.send_joint_command()
        except Exception as e:
            self.get_logger().error(f"[PACKET] Failed to send packet: {e}.")

    def send_joint_command(self) -> None:
        """
        Convert the cached command from radians to degrees, clamp to the joint limits,
        encode a 37-byte command packet, and write it to the serial port.
        """
        serial_conn = self.serial_conn
        right_arm_command = self.right_arm_command
        left_arm_command = self.left_arm_command
        neck_command = self.neck_command
        if (serial_conn is None or not serial_conn.is_open
                or right_arm_command is None or left_arm_command is None or neck_command is None):
            return

        right_arm_deg = self.clamp_and_warn(
            "right_arm", [math.degrees(v) for v in right_arm_command], self.right_arm_limits
        )
        left_arm_deg = self.clamp_and_warn(
            "left_arm", [math.degrees(v) for v in left_arm_command], self.left_arm_limits
        )
        neck_deg = self.clamp_and_warn(
            "neck", [math.degrees(v) for v in neck_command], self.neck_limit
        )

        packet = encode_joint_command(
            right_arm_deg=right_arm_deg,
            left_arm_deg=left_arm_deg,
            right_gripper_pred=self.gripper_state["right"],
            left_gripper_pred=self.gripper_state["left"],
            neck_deg=neck_deg[0],
        )

        with self.serial_lock:
            serial_conn.write(packet)
            time.sleep(0.01)

        self.get_logger().debug(
            f"[PACKET] Sent R={[round(v, 1) for v in right_arm_deg]} "
            f"L={[round(v, 1) for v in left_arm_deg]} "
            f"grip=({self.gripper_state['right']}, {self.gripper_state['left']}) "
            f"neck={neck_deg[0]:.1f}"
        )

    def clamp_and_warn(self, name: str, values: list[float], limits: list[tuple]) -> list[float]:
        """
        Clamp joint values to their limits and warn about any that were out of range.

        :param name: joint group label used in the log message.
        :param values: joint angles in degrees.
        :param limits: same-length sequence of ``(low, high)`` bounds, one per value.
        :return: the clamped values, in the same order as ``values``.
        """
        clamped = clamp_joint_angles(values, limits)
        for i, (value, clamped_value) in enumerate(zip(values, clamped)):
            if value != clamped_value:
                self.get_logger().warning(
                    f"[PACKET] {name} joint {i + 1} command {value:.1f} out of range, clamped to {clamped_value:.1f}."
                )
        return clamped

    def destroy_node(self) -> None:
        """
        Close the serial port, then destroy the node.
        """
        self.get_logger().info("[NODE] Shutting down JointCommandWriter node...")
        try:
            if self.serial_conn is not None and self.serial_conn.is_open:
                self.serial_conn.close()
                self.get_logger().info("[SERIAL] Serial port closed.")
        except Exception as e:
            self.get_logger().warning(f"[SERIAL] Failed to close serial port: {e}.")
        super().destroy_node()


def main(args=None):
    logger = get_logger("joint_command_writer_node")
    rclpy.init(args=args)
    node = None

    try:
        node = JointCommandWriterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("[NODE] Ctrl-C received, shutting down.")
    except Exception as e:
        logger.fatal(f"[NODE] Node crashed: {e}.")
    finally:
        if node is not None:
            node.destroy_node()     # close serial (guarded internally)
        if rclpy.ok():
            rclpy.shutdown()        # guard avoids double-shutdown error


if __name__ == "__main__":
    main()
