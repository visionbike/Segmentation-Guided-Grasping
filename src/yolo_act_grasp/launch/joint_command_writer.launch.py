import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory("yolo_act_grasp")
    config_dir = os.path.join(share_dir, "config")

    serial_enable = LaunchConfiguration("serial_enable")
    startup_delay_sec = LaunchConfiguration("startup_delay_sec")

    return LaunchDescription([
        DeclareLaunchArgument(
        "serial_enable",
            default_value="true",
            description="Open the serial port (false = dry run, no motor commands)",
        ),
        DeclareLaunchArgument(
            "startup_delay_sec",
            default_value="3.0",
            description="Seconds to wait before the first command packet is sent",
        ),

        LogInfo(msg="[joint_command_writer] ROBOT WILL MOVE after the startup delay."),

        Node(
            package="yolo_act_grasp",
            executable="joint_command_writer",
            name="joint_command_writer",
            output="screen",
            parameters=[
                os.path.join(config_dir, "joint_command_writer.yaml"),
                {
                    "serial_enable": serial_enable,
                    "startup_delay_sec": startup_delay_sec,
                },
            ],
        ),
    ])
