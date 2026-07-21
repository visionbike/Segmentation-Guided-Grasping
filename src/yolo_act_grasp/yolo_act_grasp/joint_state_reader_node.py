import os
import csv
import time
import serial
import numpy as np
from typing import TextIO

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from std_msgs.msg import Float32MultiArray

from .serial_protocol import Protocol, verify_checksum, compute_checksum, decode_joint_state


class JointStateReaderNode(Node):
    """
    Read the robot's joint state from the Teensy over serial and publish it.

    Inbound half of the U2D2 serial link. Polls the serial port every ``poll_interval`` seconds,
    parses the fixed-length feedback packets (header 0x3E 0xC1 0x02 0x25, 37 bytes incl. trailing
    checksum at index 36), and decodes the 17-dim proprioceptive state:

        [0:7]   right arm joints R1-R7   (radians; wire units are 0.1 deg)
        [7:14]  left arm joints L1-L7    (radians; wire units are 0.1 deg)
        [14]    right gripper status     (discrete 0/1)
        [15]    left gripper status      (discrete 0/1)
        [16]    neck joint angle         (radians)

    The decoded vector is published as ``Float32MultiArray`` on ``/joint_state_feedback`` where ``obs_sync``
    consumes it as the ``qpos`` component of the ACT policy's observation. Optionally logs each feedback frame
    to CSV when ``qpos_csv_path`` is set.

    This node closes the control loop: motors move -> encoders change -> JointStateReader republishes the new
    state for the next observation.
    """
    def __init__(self):
        super().__init__("joint_state_reader_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("com_port", "/dev/teensy_serial")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("poll_interval", 0.1)    # how often a packet receive (unit: s)
        self.declare_parameter("serial_enable", True)
        self.declare_parameter("joint_state_feedback_topic", "/joint_state_feedback")

        self.declare_parameter("log_qpos_to_csv", True)
        self.declare_parameter("qpos_csv_path", "")     # empty => CSV logging disabled

        self.com_port = self.get_parameter("com_port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.poll_interval = self.get_parameter("poll_interval").get_parameter_value().double_value
        self.serial_enable = self.get_parameter("serial_enable").get_parameter_value().bool_value
        self.joint_state_feedback_topic = self.get_parameter("joint_state_feedback_topic").get_parameter_value().string_value

        self.log_qpos_to_csv = self.get_parameter("log_qpos_to_csv").get_parameter_value().bool_value
        self.qpos_csv_path = self.get_parameter("qpos_csv_path").get_parameter_value().string_value

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------
        self.serial_conn: serial.Serial | None = None
        self.rx_index: int = 0                   # the write index into the packet buffer
        self.rx_buffer: bytearray = bytearray(64)      # the assembly buffer for the current packet
        self.right_arm_rad: np.ndarray = np.zeros(7)
        self.left_arm_rad: np.ndarray = np.zeros(7)
        self.right_gripper_status: int = 0
        self.left_gripper_status: int = 0
        self.neck_rad: np.ndarray = np.zeros(1)

        self.qpos_csv_file: TextIO | None = None
        self.qpos_csv_initialized: bool = False
        self.log_start_time: float = time.time()
        self.csv_row_index: int = 0

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.joint_state_feedback_pub = self.create_publisher(
            Float32MultiArray,
            self.joint_state_feedback_topic,
            10
        )

        # --------------------------------------------------
        # CSV initialization
        # --------------------------------------------------
        if self.log_qpos_to_csv and self.qpos_csv_path:
            self.init_qpos_csv_logging()
        else:
            self.log_qpos_to_csv = False

        # --------------------------------------------------
        # Serial initialization
        # --------------------------------------------------
        self.init_serial()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.timer = self.create_timer(self.poll_interval, self.poll_serial_callback)

        self.get_logger().info("[NODE] =========================================")
        self.get_logger().info("[NODE] JointStateReader ROS2 node started")
        self.get_logger().info(f"[NODE] com_port                   : {self.com_port}")
        self.get_logger().info(f"[NODE] baudrate                   : {self.baudrate}")
        self.get_logger().info(f"[NODE] poll_interval              : {self.poll_interval}")
        self.get_logger().info(f"[NODE] serial_enable              : {self.serial_enable}")
        self.get_logger().info(f"[NODE] joint_state_feedback_topic : {self.joint_state_feedback_topic}")
        self.get_logger().info(f"[NODE] log_qpos_to_csv            : {self.log_qpos_to_csv}")
        self.get_logger().info(f"[NODE] qpos_csv_path              : {self.qpos_csv_path}")
        self.get_logger().info("[NODE] =========================================")

    def init_qpos_csv_logging(self) -> None:
        """
        Open the qpos CSV file for writing, creating parent directories if necessary.
        """
        try:
            parent_dir = os.path.dirname(self.qpos_csv_path)
            if parent_dir != "":
                os.makedirs(parent_dir, exist_ok=True)
            self.qpos_csv_file = open(self.qpos_csv_path, "w", newline="", encoding="utf-8")
            self.get_logger().info(f"[CSV] Logging enabled: {self.qpos_csv_path}.")
        except Exception as e:
            self.get_logger().fatal(f"[CSV] Failed to open qpos CSV file {self.qpos_csv_path}: {e}.")
            raise RuntimeError(f"[CSV] Failed to open qpos CSV file: {e}.")

    def log_received_qpos(
        self,
        right_arm_rad: np.ndarray,
        left_arm_rad: np.ndarray,
        right_gripper_status: int,
        left_gripper_status: int,
        neck_rad: np.ndarray,
    ) -> None:
        """
        Append one decoded qpos frame as a row to the CSV log.

        :param right_arm_rad: 7 right-arm joint angles (radians).
        :param left_arm_rad: 7 left-arm joint angles (radians).
        :param right_gripper_status: right gripper status (0/1).
        :param left_gripper_status: left gripper status (0/1).
        :param neck_rad: 1-element array with the neck joint angle (radians).
        """
        if not self.log_qpos_to_csv or self.qpos_csv_file is None:
            return

        qpos_csv_writer = csv.writer(self.qpos_csv_file)

        # write the header once, on the first logged sample
        if not self.qpos_csv_initialized:
            header = [
                "step",
                "time_sec",
                "R1_rad", "R2_rad", "R3_rad", "R4_rad", "R5_rad", "R6_rad", "R7_rad",
                "L1_rad", "L2_rad", "L3_rad", "L4_rad", "L5_rad", "L6_rad", "L7_rad",
                "right_gripper", "left_gripper", "neck_rad"
            ]
            qpos_csv_writer.writerow(header)
            self.qpos_csv_initialized = True

        timestamp = time.time() - self.log_start_time
        row = [self.csv_row_index, timestamp]
        row += [float(v) for v in right_arm_rad]
        row += [float(v) for v in left_arm_rad]
        row += [float(right_gripper_status), float(left_gripper_status)]
        row += [float(v) for v in neck_rad]

        qpos_csv_writer.writerow(row)
        self.qpos_csv_file.flush()
        self.csv_row_index += 1

    def init_serial(self) -> None:
        """
        Open the serial port to the Teensy.
        """
        if self.serial_enable:
            try:
                self.serial_conn = serial.Serial(
                    self.com_port,
                    self.baudrate,
                    timeout=0,
                    write_timeout=1
                )
                self.get_logger().info("[SERIAL] Serial enabled.")
            except Exception as e:
                self.serial_conn = None
                self.get_logger().error(f"[SERIAL] Failed to open port {self.com_port}: {e}.")
        else:
            self.serial_conn = None
            self.get_logger().warning("[SERIAL] Serial disabled (serial_enable=False).")

    def reset_packet(self) -> None:
        """
        Reset the packet parser back to waiting for a header byte.
        """
        self.rx_index = 0

    def poll_serial_callback(self) -> None:
        """
        Read all available serial bytes in one syscall and feed each to the packet parser.
        """
        if self.serial_conn is None or not self.serial_conn.is_open:
            return

        try:
            n = self.serial_conn.in_waiting
            if not n:
                return
            for b in self.serial_conn.read(n):  # one read syscall for the whole buffer
                self._feed_byte(b)
        except Exception as e:
            self.get_logger().error(f"[PACKET] Failed to receive packet: {e}.", throttle_duration_sec=1.0)
            self.reset_packet()

    def _feed_byte(self, b: int) -> None:
        """
        Advance the header/payload/checksum state machine by one byte.

        Matches the 4-byte header (0x3E 0xC1 0x02 0x25), accumulates the payload up to the declared length
        (``rx_buffer[3]`` == 37), verifies the checksum (sum of bytes [0:36] == byte 36), and dispatches a
        complete packet to ``publish_joint_state``. Any mismatch resets the parser.

        :param b: the parsed byte.
        """
        idx = self.rx_index

        # header
        if idx == 0:
            if b == Protocol.HEADER:
                self.rx_buffer[0] = b
                self.rx_index = 1
        elif idx == 1:
            if b == Protocol.FEEDBACK_ID:
                self.rx_buffer[1] = b
                self.rx_index = 2
            else:
                self.reset_packet()
        elif idx == 2:
            if b == Protocol.FEEDBACK_SUBTYPE:
                self.rx_buffer[2] = b
                self.rx_index = 3
            else:
                self.reset_packet()
        elif idx == 3:
            if b == Protocol.PACKET_LENGTH:   # total packet length = 37 (0x25)
                self.rx_buffer[3] = b
                self.rx_index = 4
            else:
                self.reset_packet()
        # data + checksum
        elif 4 <= idx <= self.rx_buffer[3] - 1:
            self.rx_buffer[idx] = b
            self.rx_index += 1
            # decode once the full packet is in
            if self.rx_index == self.rx_buffer[3]:
                if not verify_checksum(self.rx_buffer):
                    self.get_logger().error(
                        f"[CHECKSUM] Mismatch: "
                        f"calc={compute_checksum(self.rx_buffer)} "
                        f"recv={self.rx_buffer[Protocol.CHECKSUM_INDEX]}.",
                        throttle_duration_sec=1.0,
                    )
                    self.reset_packet()
                    return
                self.publish_joint_state()
        else:
            self.reset_packet()

    def publish_joint_state(self) -> None:
        """
        Decode a verified feedback packet, publish it and log it.
        """
        try:
            joint_state = decode_joint_state(self.rx_buffer)
            self.right_arm_rad = np.deg2rad(joint_state.right_arm_deg)
            self.left_arm_rad = np.deg2rad(joint_state.left_arm_deg)
            self.right_gripper_status = joint_state.right_gripper_status
            self.left_gripper_status = joint_state.left_gripper_status
            self.neck_rad = np.deg2rad([joint_state.neck_deg])

            self.get_logger().debug(
                f"[PACKET] R={np.round(self.right_arm_rad, 4)} "
                f"L={np.round(self.left_arm_rad, 4)} "
                f"grip=({self.right_gripper_status}, {self.left_gripper_status}) "
                f"neck={self.neck_rad[0]:.4f}"
            )

            # publish
            msg = Float32MultiArray()
            msg.data = np.concatenate([
                self.right_arm_rad,
                self.left_arm_rad,
                [float(self.right_gripper_status), float(self.left_gripper_status)],
                self.neck_rad,
            ]).tolist()
            self.joint_state_feedback_pub.publish(msg)

            # log to csv
            self.log_received_qpos(
                self.right_arm_rad,
                self.left_arm_rad,
                self.right_gripper_status,
                self.left_gripper_status,
                self.neck_rad
            )
            self.reset_packet()
        except Exception as e:
            self.get_logger().error(f"[PACKET] Failed to decode joint state packet: {e}.")
            self.reset_packet()

    def destroy_node(self) -> None:
        """
        Close the serial port and CSV file, then destroy the node.
        """
        self.get_logger().info("[NODE] Shutting down JointStateReader node...")
        try:
            if self.serial_conn is not None and self.serial_conn.is_open:
                self.serial_conn.close()
                self.get_logger().info("[SERIAL] Serial port closed.")
        except Exception as e:
            self.get_logger().warning(f"[SERIAL] Failed to close serial port: {e}.")

        try:
            if self.qpos_csv_file is not None:
                self.qpos_csv_file.close()
                self.get_logger().info("[CSV] File closed.")
        except Exception as e:
            self.get_logger().warning(f"[CSV] Failed to close CSV file: {e}.")
        super().destroy_node()


def main(args=None):
    logger = get_logger("joint_state_reader_node")
    rclpy.init(args=args)
    node = None

    try:
        node = JointStateReaderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("[NODE] Ctrl-C received, shutting down.")
    except Exception as e:
        logger.fatal(f"[NODE] Node crashed: {e}.")
    finally:
        if node is not None:
            node.destroy_node()     # close serial & CSV (each guarded internally)
        if rclpy.ok():
            rclpy.shutdown()        # guard avoids double-shutdown error


if __name__ == "__main__":
    main()