#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the three RealSense cameras defined in config/cameras.yaml.

Each camera runs as its own realsense2_camera_node with
namespace = camera_name = <name>, so the color topic is:
    /<name>/<name>/color/image_raw
(matching the input topics in config/obs_sync.yaml).
"""

import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share_dir = get_package_share_directory("act_yolo_grasp")
    config_path = os.path.join(share_dir, "config", "cameras.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cameras = yaml.safe_load(f)["cameras"]

    nodes = []
    for cam in cameras:
        name = cam["name"]
        nodes.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace=name,
                name=f"{name}_realsense_node",
                output="screen",
                parameters=[
                    {
                        "serial_no": ParameterValue(str(cam["serial"]), value_type=str),
                        "camera_name": name,
                        "camera_namespace": name,
                        "enable_color": True,
                        "enable_depth": False,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_gyro": False,
                        "enable_accel": False,
                        cam["profile_param"]: cam["color_profile"],
                        "align_depth.enable": False,
                        "pointcloud.enable": False,
                    }
                ],
            )
        )

    return LaunchDescription(nodes)
