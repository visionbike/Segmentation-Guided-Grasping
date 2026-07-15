import os
from glob import glob
from setuptools import find_packages, setup

package_name = "act_yolo_grasp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felix",
    maintainer_email="phuc.visionbike@gmail.com",
    description=(
        "Segmentation-guided grasping: ACT policy driven by task-conditioned "
        "YOLO segmentation masks (OpenVINO deployment for Intel AI PC)"
    ),
    license="Apache-2.0",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
    "console_scripts": [
        "obs_sync = act_yolo_grasp.obs_sync_node:main",
        "yolo_seg = act_yolo_grasp.yolo_seg_node:main",
        "act_policy = act_yolo_grasp.act_policy_node:main",
        "visualize_image = act_yolo_grasp.visualize_node:main",
        # Enable if the U2D2 motor serial link runs on this machine
        # (copy packet_processor_{send,receive}.py from the IL package first):
        # "packet_processor_send = act_yolo_grasp.packet_processor_send:main",
        # "packet_processor_receive = act_yolo_grasp.packet_processor_receive:main",
    ]},
)
