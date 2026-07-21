import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory("yolo_act_grasp")
    config_dir = os.path.join(share_dir, "config")

    serial_enable = LaunchConfiguration("serial_enable")

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_enable",
            default_value="true",
            description="Open the serial port (false = run the node without touching hardware)",
        ),

        Node(
            package="yolo_act_grasp",
            executable="joint_state_reader",
            name="joint_state_reader",
            output="screen",
            parameters=[
                os.path.join(config_dir, "joint_state_reader.yaml"),
                {"serial_enable": serial_enable},
            ],
        ),
    ])
