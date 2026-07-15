#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import threading
import serial
import serial.tools.list_ports

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class Packet_sender(Node):
    def __init__(self):
        super().__init__("my_robot_controller_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("com_port", "/dev/teensy_serial")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("send_interval", 0.1)
        self.declare_parameter("sender_enable", True)
        self.declare_parameter("motor_speed", 5)
        self.declare_parameter("motor_action_angle_topic", "/motor_action_angle_topic")
        self.declare_parameter("stop_signal_topic", "/episode_stop")
        self.declare_parameter("start_signal_topic", "/episode_start")


        # per-hand (left/right) close threshold
        self.declare_parameter("right_gripper_threshold", 0.5)
        self.declare_parameter("left_gripper_threshold", 0.5)

        # consecutive frames required before the state switches
        self.declare_parameter("right_gripper_required_count", 1)
        self.declare_parameter("left_gripper_required_count", 1)
        self.declare_parameter("startup_delay_sec", 3.0)
        self.declare_parameter("record_begin_topic", "/episode_record_begin")
        self.record_begin_topic = self.get_parameter("record_begin_topic").get_parameter_value().string_value

        self.startup_delay_sec = self.get_parameter("startup_delay_sec").get_parameter_value().double_value

        self.com_port = self.get_parameter("com_port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.send_interval = self.get_parameter("send_interval").get_parameter_value().double_value
        self.sender_enable = self.get_parameter("sender_enable").get_parameter_value().bool_value
        self.motor_speed = self.get_parameter("motor_speed").get_parameter_value().integer_value
        self.motor_action_angle_topic = self.get_parameter("motor_action_angle_topic").get_parameter_value().string_value
        self.stop_signal_topic = self.get_parameter("stop_signal_topic").get_parameter_value().string_value

        self.right_gripper_threshold = self.get_parameter("right_gripper_threshold").get_parameter_value().double_value
        self.left_gripper_threshold = self.get_parameter("left_gripper_threshold").get_parameter_value().double_value
        self.right_gripper_required_count = self.get_parameter("right_gripper_required_count").get_parameter_value().integer_value
        self.left_gripper_required_count = self.get_parameter("left_gripper_required_count").get_parameter_value().integer_value
        self.start_signal_topic = self.get_parameter("start_signal_topic").get_parameter_value().string_value

        if self.right_gripper_required_count < 1:
            self.right_gripper_required_count = 1
        if self.left_gripper_required_count < 1:
            self.left_gripper_required_count = 1

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------
        self.lock = threading.Lock()
        self.U2D2 = None
        self.R_arm_angle_action = None
        self.L_arm_angle_action = None
        self.R_gripper_action = None
        self.L_gripper_action = None
        self.Neck_action = None

        self.stop_requested = False

        # gripper state (the 0/1 actually sent out)
        self.R_gripper_state = 0
        self.L_gripper_state = 0

        # gripper consecutive-frame counters
        self.R_gripper_high_count = 0
        self.R_gripper_low_count = 0
        self.L_gripper_high_count = 0
        self.L_gripper_low_count = 0

        self.right_arm_limits = [
            (-90.0, 90.0),    # R1      
            (-90.0, 5.0),    # R2
            (-90.0, 90.0),  # R3
            (-0.0, 150.0),    # R4
            (-90.0, 90.0),  # R5
            (-15.0, 15.0),
            (-30.0, 30.0),
        ]

        self.left_arm_limits = [
            (-90.0, 90.0),    # L1
            (-5.0, 90.0),    # L2
            (-90.0, 90.0),  # L3
            (-150.0, 0.0),    # L4
            (-90.0, 90.0),  # L5
            (-15.0, 15.0),
            (-30.0, 30.0),
        ]

        self.Neck_limit = [
            (-90.0, 90.0)
        ]

        self.start_time = time.time()
        self.send_enable_after_time = self.start_time + self.startup_delay_sec
        self.last_countdown_print_sec = None

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.action_angle_sub = self.create_subscription(
            Float32MultiArray,
            self.motor_action_angle_topic,
            self.action_angle_callback,
            10
        )

        self.stop_signal_sub = self.create_subscription(
            Bool,
            self.stop_signal_topic,
            self.stop_signal_callback,
            10
        )

        start_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.start_pub = self.create_publisher(Bool, self.start_signal_topic, start_qos)
        self.start_sent = False
        self.record_begin_pub = self.create_publisher(Bool, self.record_begin_topic, 10)
        self.record_begin_sent = False

        # --------------------------------------------------
        # Serial init
        # --------------------------------------------------
        self.controller_init()
        self.publish_start_signal_once()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.timer = self.create_timer(self.send_interval, self.send_command_callback)

        self.get_logger().info("=========================================")
        self.get_logger().info("Packet Sender ROS2 node started")
        self.get_logger().info(f"com_port                      : {self.com_port}")
        self.get_logger().info(f"baudrate                      : {self.baudrate}")
        self.get_logger().info(f"send_interval                 : {self.send_interval}")
        self.get_logger().info(f"sender_enable                 : {self.sender_enable}")
        self.get_logger().info(f"motor_action_angle_topic      : {self.motor_action_angle_topic}")
        self.get_logger().info(f"stop_signal_topic             : {self.stop_signal_topic}")
        self.get_logger().info(f"right_gripper_threshold       : {self.right_gripper_threshold}")
        self.get_logger().info(f"left_gripper_threshold        : {self.left_gripper_threshold}")
        self.get_logger().info(f"right_gripper_required_count  : {self.right_gripper_required_count}")
        self.get_logger().info(f"left_gripper_required_count   : {self.left_gripper_required_count}")
        self.get_logger().info(f"start_signal_topic            : {self.start_signal_topic}")
        self.get_logger().info("=========================================")

    # --------------------------------------------------
    # Serial init
    # --------------------------------------------------
    def controller_init(self):
        if self.sender_enable:
            try:
                self.U2D2 = serial.Serial(
                    self.com_port,
                    self.baudrate,
                    timeout=1,
                    write_timeout=1
                )
                self.get_logger().info("[PACKET INFO] ROBOT CONTROL ENABLE")
            except Exception:
                self.U2D2 = None
                self.get_logger().error("[PACKET INFO] ROBOT CONTROL DISABLE")
        else:
            self.U2D2 = None
            self.get_logger().warning("[PACKET INFO] ROBOT CONTROL DISABLE")

    # --------------------------------------------------
    # Stop callback
    # --------------------------------------------------
    def stop_signal_callback(self, msg: Bool):
        if not msg.data:
            return

        if self.stop_requested:
            return

        self.stop_requested = True
        self.get_logger().warning("[STOP SIGNAL] Received True. Stopping sender node...")

        try:
            self.timer.cancel()
        except Exception:
            pass

        try:
            if self.U2D2 is not None and self.U2D2.is_open:
                self.U2D2.close()
                self.get_logger().info("Serial port closed by stop signal")
        except Exception as e:
            self.get_logger().warning(f"Failed to close serial port on stop: {e}")

        rclpy.shutdown()

    # --------------------------------------------------
    # Subscribers callback
    # --------------------------------------------------
    def action_angle_callback(self, msg: Float32MultiArray):
        if self.stop_requested:
            return

        data = list(msg.data)

        # [L1..L7, R1..R7, L_gripper, R_gripper, Neck]
        if len(data) < 17:
            return

        self.L_arm_angle_action = data[0:7]
        self.R_arm_angle_action = data[7:14]
        self.L_gripper_action = float(data[14])
        self.R_gripper_action = float(data[15])
        self.Neck_action = [data[16]]

        # update gripper state on every new message
        self.update_right_gripper_state(self.R_gripper_action)
        self.update_left_gripper_state(self.L_gripper_action)

    def publish_start_signal_once(self):
        start_msg = Bool()
        start_msg.data = True

        for _ in range(5):
            self.start_pub.publish(start_msg)
            time.sleep(0.05)

        self.get_logger().info(
            f"[START SIGNAL] Published /episode_start = True, "
            f"will start sending packets after {self.startup_delay_sec:.1f} sec"
        )

    # --------------------------------------------------
    # Gripper debounce / consecutive-count logic
    # --------------------------------------------------
    def update_right_gripper_state(self, value):
        if value >= self.right_gripper_threshold:
            self.R_gripper_high_count += 1
            self.R_gripper_low_count = 0
        else:
            self.R_gripper_low_count += 1
            self.R_gripper_high_count = 0

        if self.R_gripper_high_count >= self.right_gripper_required_count:
            if self.R_gripper_state != 1:
                self.get_logger().info(
                    f"[R_GRIPPER] state 0 -> 1 | value={value:.3f}, "
                    f"high_count={self.R_gripper_high_count}"
                )
            self.R_gripper_state = 1

        elif self.R_gripper_low_count >= self.right_gripper_required_count:
            if self.R_gripper_state != 0:
                self.get_logger().info(
                    f"[R_GRIPPER] state 1 -> 0 | value={value:.3f}, "
                    f"low_count={self.R_gripper_low_count}"
                )
            self.R_gripper_state = 0

    def update_left_gripper_state(self, value):
        if value >= self.left_gripper_threshold:
            self.L_gripper_high_count += 1
            self.L_gripper_low_count = 0
        else:
            self.L_gripper_low_count += 1
            self.L_gripper_high_count = 0

        if self.L_gripper_high_count >= self.left_gripper_required_count:
            if self.L_gripper_state != 1:
                self.get_logger().info(
                    f"[L_GRIPPER] state 0 -> 1 | value={value:.3f}, "
                    f"high_count={self.L_gripper_high_count}"
                )
            self.L_gripper_state = 1

        elif self.L_gripper_low_count >= self.left_gripper_required_count:
            if self.L_gripper_state != 0:
                self.get_logger().info(
                    f"[L_GRIPPER] state 1 -> 0 | value={value:.3f}, "
                    f"low_count={self.L_gripper_low_count}"
                )
            self.L_gripper_state = 0

    # --------------------------------------------------
    # Timer callback
    # --------------------------------------------------
    def send_command_callback(self):
        if self.stop_requested:
            return

        now = time.time()
        if now < self.send_enable_after_time:
            remain = int(math.ceil(self.send_enable_after_time - now))

            if self.last_countdown_print_sec != remain:
                self.get_logger().info(
                    f"[STARTUP DELAY] Sending will start in {remain} second(s)..."
                )
                self.last_countdown_print_sec = remain

            return
        
        if not self.record_begin_sent:
            msg = Bool()
            msg.data = True
            for _ in range(5):
                self.record_begin_pub.publish(msg)
                time.sleep(0.02)
            self.record_begin_sent = True
            self.get_logger().info("[RECORD BEGIN] Published /episode_record_begin = True")

        if self.U2D2 is None or not self.U2D2.is_open:
            return
        try:
            R_arm = self.R_arm_angle_action
            L_arm = self.L_arm_angle_action
            Neck = self.Neck_action

            if R_arm is not None and L_arm is not None and Neck is not None and \
               self.R_gripper_action is not None and self.L_gripper_action is not None:

                R_gripper = self.R_gripper_state
                L_gripper = self.L_gripper_state

                self.dual_arm_angle_cmd_with_P_control(R_arm, L_arm, R_gripper, L_gripper, Neck)

                # self.get_logger().info(
                #     f"[GRIPPER STATUS] "
                #     f"R_raw={self.R_gripper_action:.3f}, R_state={R_gripper}, "
                #     f"R_high={self.R_gripper_high_count}, R_low={self.R_gripper_low_count} | "
                #     f"L_raw={self.L_gripper_action:.3f}, L_state={L_gripper}, "
                #     f"L_high={self.L_gripper_high_count}, L_low={self.L_gripper_low_count}"
                # )
            else:
                self.get_logger().warning("[PACKET WARN] No arms data")
        except Exception as e:
            self.get_logger().error(f"[PACKET ERROR] fail to send packet: {e}")

    # --------------------------------------------------
    # Packet functions
    # --------------------------------------------------
    def dual_arm_angle_cmd(self, R, L, S):
        # radian to angle
        R = [math.degrees(x) for x in R]
        L = [math.degrees(x) for x in L]

        # angle protection
        R = self.clamp_joint_angle(R, self.right_arm_limits, arm_name="right_arm")
        L = self.clamp_joint_angle(L, self.left_arm_limits, arm_name="left_arm")

        # Right arm angle
        R1 = int(R[0] * 10)
        R2 = int(R[1] * 10)
        R3 = int(R[2] * 10)
        R4 = int(R[3] * 10)
        R5 = int(R[4] * 10)
        R6 = int(R[5] * 10)
        R7 = int(R[6] * 10)

        # Left arm angle
        L1 = int(L[0] * 10)
        L2 = int(L[1] * 10)
        L3 = int(L[2] * 10)
        L4 = int(L[3] * 10)
        L5 = int(L[4] * 10)
        L6 = int(L[5] * 10)
        L7 = int(L[6] * 10)

        # Speed
        speed = int(S * 10)

        # Packet
        buffer = bytearray(33)
        buffer[0] = 0x3E
        buffer[1] = 0xB1
        buffer[2] = 0x00
        buffer[3] = 0x21

        # Right arm
        buffer[4] = (R1 >> 8) & 0xFF
        buffer[5] = (R1 >> 0) & 0xFF
        buffer[6] = (R2 >> 8) & 0xFF
        buffer[7] = (R2 >> 0) & 0xFF
        buffer[8] = (R3 >> 8) & 0xFF
        buffer[9] = (R3 >> 0) & 0xFF
        buffer[10] = (R4 >> 8) & 0xFF
        buffer[11] = (R4 >> 0) & 0xFF
        buffer[12] = (R5 >> 8) & 0xFF
        buffer[13] = (R5 >> 0) & 0xFF
        buffer[14] = (R6 >> 8) & 0xFF
        buffer[15] = (R6 >> 0) & 0xFF
        buffer[16] = (R7 >> 8) & 0xFF
        buffer[17] = (R7 >> 0) & 0xFF


        # Left arm
        buffer[18] = (L1 >> 8) & 0xFF
        buffer[19] = (L1 >> 0) & 0xFF
        buffer[20] = (L2 >> 8) & 0xFF
        buffer[21] = (L2 >> 0) & 0xFF
        buffer[22] = (L3 >> 8) & 0xFF
        buffer[23] = (L3 >> 0) & 0xFF
        buffer[24] = (L4 >> 8) & 0xFF
        buffer[25] = (L4 >> 0) & 0xFF
        buffer[26] = (L5 >> 8) & 0xFF
        buffer[27] = (L5 >> 0) & 0xFF
        buffer[28] = (L5 >> 8) & 0xFF
        buffer[29] = (L5 >> 0) & 0xFF
        buffer[30] = (L5 >> 8) & 0xFF
        buffer[31] = (L5 >> 0) & 0xFF
        buffer[32] = (speed >> 0) & 0xFF

        with self.lock:
            self.U2D2.write(buffer)
            time.sleep(0.01)

        self.get_logger().info(
            f"Dual arm cmd sent | "
            f"Right: [{R1}, {R2}, {R3}, {R4}, {R5}, {R6}, {R7}] | "
            f"Left: [{L1}, {L2}, {L3}, {L4}, {L5}, {L6}, {L7}] | "
            f"Speed: {speed}"
        )

    def dual_arm_angle_cmd_with_P_control(self, R, L, R_gripper, L_gripper, N):
        # radian to angle
        R = [math.degrees(x) for x in R]
        L = [math.degrees(x) for x in L]
        N = [math.degrees(x) for x in N]

        # angle protection
        R = self.clamp_joint_angle(R, self.right_arm_limits, arm_name="right_arm")
        L = self.clamp_joint_angle(L, self.left_arm_limits, arm_name="left_arm")
        N = self.clamp_joint_angle(N, self.Neck_limit, arm_name="Neck")

        # angle -> 0.1 deg integer
        R1 = int(round(R[0] * 10))
        R2 = int(round(R[1] * 10))
        R3 = int(round(R[2] * 10))
        R4 = int(round(R[3] * 10))
        R5 = int(round(R[4] * 10))
        R6 = int(round(R[5] * 10))
        R7 = int(round(R[6] * 10))

        L1 = int(round(L[0] * 10))
        L2 = int(round(L[1] * 10))
        L3 = int(round(L[2] * 10))
        L4 = int(round(L[3] * 10))
        L5 = int(round(L[4] * 10))
        L6 = int(round(L[5] * 10))
        L7 = int(round(L[6] * 10))

        Neck = int(round(N[0] * 10))

        # force gripper to 0/1
        Rg = 1 if float(R_gripper) > 0.5 else 0
        Lg = 1 if float(L_gripper) > 0.5 else 0

        # Packet: 34 data bytes + 1 checksum = 35 bytes
        buffer = bytearray(37)
        buffer[0] = 0x3E
        buffer[1] = 0xB2
        buffer[2] = 0x00
        buffer[3] = 0x25   # 35 bytes

        # Right arm
        buffer[4]  = (R1 >> 8) & 0xFF
        buffer[5]  = R1 & 0xFF
        buffer[6]  = (R2 >> 8) & 0xFF
        buffer[7]  = R2 & 0xFF
        buffer[8]  = (R3 >> 8) & 0xFF
        buffer[9]  = R3 & 0xFF
        buffer[10] = (R4 >> 8) & 0xFF
        buffer[11] = R4 & 0xFF
        buffer[12] = (R5 >> 8) & 0xFF
        buffer[13] = R5 & 0xFF
        buffer[14] = (R6 >> 8) & 0xFF
        buffer[15] = R6 & 0xFF
        buffer[16] = (R7 >> 8) & 0xFF
        buffer[17] = R7 & 0xFF

        # Left arm
        buffer[18] = (L1 >> 8) & 0xFF
        buffer[19] = L1 & 0xFF
        buffer[20] = (L2 >> 8) & 0xFF
        buffer[21] = L2 & 0xFF
        buffer[22] = (L3 >> 8) & 0xFF
        buffer[23] = L3 & 0xFF
        buffer[24] = (L4 >> 8) & 0xFF
        buffer[25] = L4 & 0xFF
        buffer[26] = (L5 >> 8) & 0xFF
        buffer[27] = L5 & 0xFF
        buffer[28] = (L6 >> 8) & 0xFF
        buffer[29] = L6 & 0xFF
        buffer[30] = (L7 >> 8) & 0xFF
        buffer[31] = L7 & 0xFF

        # gripper
        buffer[32] = Rg
        buffer[33] = Lg

        # Neck
        buffer[34] = (Neck >> 8) & 0xFF
        buffer[35] = Neck & 0xFF

        # checksum = sum of the preceding bytes & 0xFF
        checksum = sum(buffer[:36]) & 0xFF
        buffer[36] = checksum

        with self.lock:
            self.U2D2.write(buffer)
            time.sleep(0.01)

        self.get_logger().info(
            f"Dual arm cmd sent (P control) | "
            f"Right: [{R1}, {R2}, {R3}, {R4}, {R5}, {R6}, {R7}, {Rg}] | "
            f"Left: [{L1}, {L2}, {L3}, {L4}, {L5}, {L6}, {L7}, {Lg}] | "
            f"Neck: [{Neck}] | checksum={checksum}"
        )

    # --------------------------------------------------
    # Angle protection
    # --------------------------------------------------
    def clamp_joint_angle(self, values, limits, arm_name="arm"):
        if len(values) != len(limits):
            raise ValueError(
                f"{arm_name} values length {len(values)} != limits length {len(limits)}"
            )
        clamped = []
        for i, (v, (low, high)) in enumerate(zip(values, limits)):
            v_clamped = max(low, min(high, v))
            if v != v_clamped:
                self.get_logger().warning(
                    f"{arm_name} joint {i+1} command {v} out of range, clamped to {v_clamped}"
                )
            clamped.append(v_clamped)
        return clamped

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down controller node...")
        try:
            if self.U2D2 is not None and self.U2D2.is_open:
                self.U2D2.close()
                self.get_logger().info("Serial port closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close serial port: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Packet_sender()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Node crashed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()