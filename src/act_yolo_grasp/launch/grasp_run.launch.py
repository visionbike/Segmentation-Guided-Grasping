#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full segmentation-guided grasping pipeline.

    cameras (3x RealSense)
        -> obs_sync   (/sync/{cam}/image_raw + /sync/qpos)
        -> yolo_seg   (/{cam}/YOLO_mask, OpenVINO on Intel GPU)
        -> act_policy (/motor_action_angle_topic, OpenVINO)
        -> visualize  (optional, use_viz:=false to disable)

External inputs the pipeline waits on:
    /motor_angle_feedback_topic  (17-dim encoder state; without it obs_sync
                                  publishes nothing)
    /llm_state                   (6-bit task state; idle pose until received)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory("act_yolo_grasp")
    launch_dir = os.path.join(share_dir, "launch")
    config_dir = os.path.join(share_dir, "config")

    use_cameras = LaunchConfiguration("use_cameras")
    use_viz = LaunchConfiguration("use_viz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_cameras",
            default_value="true",
            description="Launch the RealSense cameras (false when replaying bags)",
        ),
        DeclareLaunchArgument(
            "use_viz",
            default_value="true",
            description="Launch the mask visualization window",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, "cameras.launch.py")
            ),
            condition=IfCondition(use_cameras),
        ),

        Node(
            package="act_yolo_grasp",
            executable="obs_sync",
            name="raw_observation_sync_node",
            output="screen",
            parameters=[os.path.join(config_dir, "obs_sync.yaml")],
        ),

        Node(
            package="act_yolo_grasp",
            executable="yolo_seg",
            name="yolo_seg",
            output="screen",
            parameters=[os.path.join(config_dir, "yolo_seg.yaml")],
        ),

        Node(
            package="act_yolo_grasp",
            executable="act_policy",
            name="act_policy_inference_node",
            output="screen",
            parameters=[os.path.join(config_dir, "act_policy.yaml")],
        ),

        Node(
            package="act_yolo_grasp",
            executable="visualize_image",
            name="multi_camera_viewer",
            output="screen",
            condition=IfCondition(use_viz),
        ),
    ])
