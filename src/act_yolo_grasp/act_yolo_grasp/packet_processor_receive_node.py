#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import struct
import serial
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Packet_receiver(Node):
    def __init__(self):
        super().__init__("packet_receiver_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("com_port", "/dev/teensy_serial")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("receive_interval", 0.1)
        self.declare_parameter("receiver_enable", True)
        self.declare_parameter("motor_angle_feedback_topic", "/motor_angle_feedback_topic")

        # CSV logging parameters
        self.declare_parameter("log_angles_to_csv", True)
        self.declare_parameter("angle_csv_path", "")  # empty => CSV logging disabled

        self.com_port = self.get_parameter("com_port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.receive_interval = self.get_parameter("receive_interval").get_parameter_value().double_value
        self.receiver_enable = self.get_parameter("receiver_enable").get_parameter_value().bool_value
        self.motor_angle_feedback_topic = self.get_parameter("motor_angle_feedback_topic").get_parameter_value().string_value

        self.log_angles_to_csv = self.get_parameter("log_angles_to_csv").get_parameter_value().bool_value
        self.angle_csv_path = self.get_parameter("angle_csv_path").get_parameter_value().string_value

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------
        self.U2D2 = None
        self.count_rx = 0
        self.readin = bytearray(64)
        self.RightArmAngle = [0.0] * 7
        self.LeftArmAngle = [0.0] * 7
        self.RightGripperStatus = 0
        self.LeftGripperStatus = 0
        self.SamplingTime = 10000
        self.MicrosecondsPerSec = 1000000

        # CSV logging state
        self.angle_csv_file = None
        self.angle_csv_writer = None
        self.angle_csv_initialized = False
        self.start_time = time.time()
        self.feedback_step = 0

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.motor_angle_feedback_pub = self.create_publisher(
            Float32MultiArray,
            self.motor_angle_feedback_topic,
            10
        )

        # --------------------------------------------------
        # CSV init
        # --------------------------------------------------
        if self.log_angles_to_csv and self.angle_csv_path:
            self.init_angle_csv_logging()
        else:
            self.log_angles_to_csv = False

        # --------------------------------------------------
        # Serial init
        # --------------------------------------------------
        self.controller_init()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.timer = self.create_timer(self.receive_interval, self.receive_callback)

        self.get_logger().info("=========================================")
        self.get_logger().info("=========================================")
        self.get_logger().info("Packet Receiver ROS2 node started")
        self.get_logger().info(f"com_port                   : {self.com_port}")
        self.get_logger().info(f"baudrate                   : {self.baudrate}")
        self.get_logger().info(f"receive_interval           : {self.receive_interval}")
        self.get_logger().info(f"receiver_enable            : {self.receiver_enable}")
        self.get_logger().info(f"motor_angle_feedback_topic : {self.motor_angle_feedback_topic}")
        self.get_logger().info(f"log_angles_to_csv          : {self.log_angles_to_csv}")
        self.get_logger().info(f"angle_csv_path             : {self.angle_csv_path}")
        self.get_logger().info("=========================================")

    # --------------------------------------------------
    # CSV logging init
    # --------------------------------------------------
    def init_angle_csv_logging(self):
        try:
            parent_dir = os.path.dirname(self.angle_csv_path)
            if parent_dir != "":
                os.makedirs(parent_dir, exist_ok=True)

            self.angle_csv_file = open(self.angle_csv_path, "w", newline="", encoding="utf-8")
            self.get_logger().info(f"Angle CSV logging enabled: {self.angle_csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to open angle CSV file: {e}")

    def write_angle_csv_header_if_needed(self):
        if self.angle_csv_initialized:
            return

        if self.angle_csv_file is None:
            return

        header = [
            "step",
            "time_sec",
            "R1_rad", "R2_rad", "R3_rad", "R4_rad", "R5_rad", "R6_rad", "R7_rad",
            "L1_rad", "L2_rad", "L3_rad", "L4_rad", "L5_rad", "L6_rad", "L7_rad"
        ]

        self.angle_csv_writer = csv.writer(self.angle_csv_file)
        self.angle_csv_writer.writerow(header)
        self.angle_csv_file.flush()
        self.angle_csv_initialized = True

    def log_received_angles(self, right_arm_angles, left_arm_angles):
        if not self.log_angles_to_csv or self.angle_csv_file is None:
            return

        self.write_angle_csv_header_if_needed()

        timestamp = time.time() - self.start_time
        row = [self.feedback_step, timestamp]
        row += [float(v) for v in right_arm_angles]
        row += [float(v) for v in left_arm_angles]

        self.angle_csv_writer.writerow(row)
        self.angle_csv_file.flush()
        self.feedback_step += 1

    # --------------------------------------------------
    # Serial init
    # --------------------------------------------------
    def controller_init(self):
        if self.receiver_enable:
            try:
                self.U2D2 = serial.Serial(
                    self.com_port,
                    self.baudrate,
                    timeout=0,
                    write_timeout=1
                )
                self.get_logger().info("[PACKET INFO] ROBOT RECEIVER ENABLE")
            except Exception as e:
                self.U2D2 = None
                self.get_logger().error("[PACKET INFO] ROBOT RECEIVER DISABLE")
                self.get_logger().error(f"[PACKET ERROR] {e}")
        else:
            self.U2D2 = None
            self.get_logger().warning("[PACKET INFO] ROBOT RECEIVER DISABLE")

    # --------------------------------------------------
    # Packet reset
    # --------------------------------------------------
    def reset_packet(self):
        self.count_rx = 0

    # --------------------------------------------------
    # Timer callback
    # --------------------------------------------------
    def receive_callback(self):
        if self.U2D2 is None or not self.U2D2.is_open:
            return

        try:
            while self.U2D2.in_waiting:
                temp = self.U2D2.read(1)
                if not temp:
                    break

                b = temp[0]

                # Header
                if b == 0x3E and self.count_rx == 0:
                    self.readin[self.count_rx] = b
                    self.count_rx += 1

                elif self.count_rx == 1:
                    if b == 0xC1:
                        self.readin[self.count_rx] = b
                        self.count_rx += 1
                    else:
                        self.reset_packet()

                elif self.count_rx == 2:
                    if b == 0x02:
                        self.readin[self.count_rx] = b
                        self.count_rx += 1
                    else:
                        self.reset_packet()

                elif self.count_rx == 3:
                    if b == 0x25:   # total packet length = 35
                        self.readin[self.count_rx] = b
                        self.count_rx += 1
                    else:
                        self.reset_packet()

                # Data + checksum
                elif 4 <= self.count_rx <= self.readin[3] - 1:
                    self.readin[self.count_rx] = b
                    self.count_rx += 1

                    # Decode
                    if self.count_rx == self.readin[3]:
                        # checksum verify
                        checksum_calc = sum(self.readin[:36]) & 0xFF        # (dimension change)
                        checksum_recv = self.readin[36]

                        if checksum_calc != checksum_recv:
                            self.get_logger().error(
                                f"[CHECKSUM ERROR] calc={checksum_calc} recv={checksum_recv}"
                            )
                            self.reset_packet()
                            return

                        if self.readin[1] == 0xC1:
                            self.motor_angle_feedback()
                        else:
                            self.reset_packet()
                else:
                    self.reset_packet()

        except Exception as e:
            self.get_logger().error(f"[PACKET ERROR] fail to receive packet: {e}")
            self.reset_packet()

    # --------------------------------------------------
    # Packet handle
    # --------------------------------------------------
    def motor_angle_feedback(self):
        try:
            if self.readin[2] == 0x02: 
                R1 = struct.unpack(">h", bytes([self.readin[4], self.readin[5]]))[0] / 10.0
                R2 = struct.unpack(">h", bytes([self.readin[6], self.readin[7]]))[0] / 10.0
                R3 = struct.unpack(">h", bytes([self.readin[8], self.readin[9]]))[0] / 10.0
                R4 = struct.unpack(">h", bytes([self.readin[10], self.readin[11]]))[0] / 10.0
                R5 = struct.unpack(">h", bytes([self.readin[12], self.readin[13]]))[0] / 10.0
                R6 = struct.unpack(">h", bytes([self.readin[14], self.readin[15]]))[0] / 10.0
                R7 = struct.unpack(">h", bytes([self.readin[16], self.readin[17]]))[0] / 10.0

                L1 = struct.unpack(">h", bytes([self.readin[18], self.readin[19]]))[0] / 10.0
                L2 = struct.unpack(">h", bytes([self.readin[20], self.readin[21]]))[0] / 10.0
                L3 = struct.unpack(">h", bytes([self.readin[22], self.readin[23]]))[0] / 10.0
                L4 = struct.unpack(">h", bytes([self.readin[24], self.readin[25]]))[0] / 10.0
                L5 = struct.unpack(">h", bytes([self.readin[26], self.readin[27]]))[0] / 10.0
                L6 = struct.unpack(">h", bytes([self.readin[28], self.readin[29]]))[0] / 10.0
                L7 = struct.unpack(">h", bytes([self.readin[30], self.readin[31]]))[0] / 10.0

                self.RightGripperStatus = self.readin[32]
                self.LeftGripperStatus = self.readin[33]

                Neck = struct.unpack(">h", bytes([self.readin[34], self.readin[35]]))[0] / 10.0

                self.RightArmAngle = np.deg2rad([R1, R2, R3, R4, R5, R6, R7])
                self.LeftArmAngle = np.deg2rad([L1, L2, L3, L4, L5, L6, L7])
                self.NeckAngle = np.deg2rad([Neck])
                self.get_logger().info("[Received] Motor angle feedback   ----> ")
                self.get_logger().info(f"{R1:.4f} {R2:.4f} {R3:.4f} {R4:.4f} {R5:.4f} {R6:.4f} {R7:.4f} {self.RightGripperStatus}")
                self.get_logger().info(f"{L1:.4f} {L2:.4f} {L3:.4f} {L4:.4f} {L5:.4f} {L6:.4f} {L7:.4f} {self.LeftGripperStatus}")
                self.get_logger().info(f"{Neck:.4f}")

                # publish
                msg = Float32MultiArray()
                msg.data = np.concatenate([self.RightArmAngle, self.LeftArmAngle, np.array([self.RightGripperStatus]), np.array([self.LeftGripperStatus]), self.NeckAngle], 0).tolist()
                # print(msg.data)
                self.motor_angle_feedback_pub.publish(msg)

                # log to csv
                self.log_received_angles(
                    right_arm_angles=self.RightArmAngle,
                    left_arm_angles=self.LeftArmAngle
                )

            self.reset_packet()

        except Exception as e:
            self.get_logger().error(f"[PACKET ERROR] fail to decode Motor angle feedback packet: {e}")
            self.reset_packet()

    

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down receiver node...")
        try:
            if self.U2D2 is not None and self.U2D2.is_open:
                self.U2D2.close()
                self.get_logger().info("Serial port closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close serial port: {e}")

        try:
            if self.angle_csv_file is not None:
                self.angle_csv_file.close()
                self.get_logger().info("Angle CSV file closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close angle CSV file: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Packet_receiver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Node crashed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()