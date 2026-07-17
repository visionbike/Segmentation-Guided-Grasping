"""Pure task/geometry logic for segmentation-guided grasping.

ROS-free helpers extracted from yolo_seg_node.py so they can be unit-tested
with synthetic masks and reused outside the node:

  - basket / object relabeling by camera view (mid/right disambiguation)
  - deterministic instance selection per camera rule
  - mask geometry: in-basket overlap test, basket-side selection,
    convex-hull basket cover mask

Stateful task orchestration (LLM state machine, instance locking/tracking,
action-done publication, mask coloring) stays in YoloSegNode.
"""

from operator import itemgetter

import cv2
import numpy as np


def relabel_basket_instances(basket_instances, camera_name,
                         camera_basket_label_map, camera_single_basket_label_map):
    """
    Relabel YOLO-detected basket instances as:
        mid_basket / right_basket / unknown_basket

    Originally YOLO outputs a single class:
        basket, basket

    Here they are converted by camera view and x position into:
        mid_basket, right_basket
    """

    if len(basket_instances) == 0:
        return []

    sorted_baskets = sorted(
        basket_instances,
        key=itemgetter("cx")
    )

    labeled_baskets = []

    # =====================================================
    # Case 1: only one basket visible.
    # Do not guess from the current task, to avoid left_wrist mislabeling mid as right.
    # =====================================================
    if len(sorted_baskets) == 1:
        item = sorted_baskets[0]

        label = camera_single_basket_label_map.get(
            camera_name,
            "unknown_basket"
        )

        labeled_baskets.append({
            "cls_name": label,
            "mask": item["mask"],
            "cx": item["cx"],
        })

        # self.get_logger().info(
        #     f"[BASKET RELABEL SINGLE] camera={camera_name}, "
        #     f"cx={item['cx']:.1f}, label={label}"
        # )

        return labeled_baskets

    # =====================================================
    # Case 2: two or more baskets visible.
    # Label only the smallest-x and largest-x baskets.
    # =====================================================
    label_rule = camera_basket_label_map.get(
        camera_name,
        {
            "left_label": "mid_basket",
            "right_label": "right_basket",
        }
    )

    left_item = sorted_baskets[0]
    right_item = sorted_baskets[-1]

    labeled_baskets.append({
        "cls_name": label_rule["left_label"],
        "mask": left_item["mask"],
        "cx": left_item["cx"],
    })

    labeled_baskets.append({
        "cls_name": label_rule["right_label"],
        "mask": right_item["mask"],
        "cx": right_item["cx"],
    })

    # self.get_logger().info(
    #     f"[BASKET RELABEL] camera={camera_name}, "
    #     f"cx={[round(item['cx'], 1) for item in sorted_baskets]}, "
    #     f"left_label={label_rule['left_label']}, "
    #     f"right_label={label_rule['right_label']}"
    # )

    return labeled_baskets

def relabel_object_instances(object_instances, camera_name,
                         camera_object_label_map, camera_single_object_label_map):
    """
    Relabel multiple instances of the same object class as:
        mid_object / right_object / unknown_object

    object_instances:
        [
            {"cls_name": "lucky", "mask": mask, "cx": cx},
            {"cls_name": "lucky", "mask": mask, "cx": cx},
        ]

    Return format:
        [
            {"cls_name": "lucky", "position_label": "mid_object", ...},
            {"cls_name": "lucky", "position_label": "right_object", ...},
        ]
    """

    if len(object_instances) == 0:
        return []

    sorted_objects = sorted(
        object_instances,
        key=itemgetter("cx")
    )

    labeled_objects = []

    # =====================================================
    # Case 1: only one instance of this class.
    # Keep it directly without guessing mid/right.
    # =====================================================
    if len(sorted_objects) == 1:
        item = sorted_objects[0]

        label = camera_single_object_label_map.get(
            camera_name,
            "unknown_object"
        )

        labeled_objects.append({
            "cls_name": item["cls_name"],
            "position_label": label,
            "mask": item["mask"],
            "cx": item["cx"],
            "cy": item["cy"],
        })

        # self.get_logger().info(
        #     f"[OBJECT RELABEL SINGLE] camera={camera_name}, "
        #     f"cls={item['cls_name']}, cx={item['cx']:.1f}, label={label}"
        # )

        return labeled_objects

    # =====================================================
    # Case 2: two or more instances of the same class.
    # Label only the smallest-x and largest-x instances.
    # =====================================================
    label_rule = camera_object_label_map.get(
        camera_name,
        {
            "left_label": "right_object",
            "right_label": "mid_object",
        }
    )

    left_item = sorted_objects[0]
    right_item = sorted_objects[-1]

    labeled_objects.append({
        "cls_name": left_item["cls_name"],
        "position_label": label_rule["left_label"],
        "mask": left_item["mask"],
        "cx": left_item["cx"],
        "cy": left_item["cy"],
    })

    labeled_objects.append({
        "cls_name": right_item["cls_name"],
        "position_label": label_rule["right_label"],
        "mask": right_item["mask"],
        "cx": right_item["cx"],
        "cy": right_item["cy"],
    })

    # self.get_logger().info(
    #     f"[OBJECT RELABEL] camera={camera_name}, "
    #     f"cls={left_item['cls_name']}, "
    #     f"cx={[round(item['cx'], 1) for item in sorted_objects]}, "
    #     f"left_label={label_rule['left_label']}, "
    #     f"right_label={label_rule['right_label']}"
    # )

    return labeled_objects

def select_object_instance_by_camera_x(instances, camera_name,
                                       camera_object_select_rule):
    """
    When multiple instances of the same class appear, pick exactly one by camera rule.

    Rules:
        top         -> larger x
        front       -> smaller x
        left_wrist  -> larger x
        right_wrist -> larger x
    """

    if len(instances) == 0:
        return []

    if len(instances) == 1:
        return [instances[0]]

    rule = camera_object_select_rule.get(camera_name, "max_x")

    if rule == "min_x":
        selected = min(instances, key=itemgetter("cx"))
    else:
        selected = max(instances, key=itemgetter("cx"))

    # self.get_logger().info(
    #     f"[OBJECT SELECT BY X] camera={camera_name}, "
    #     f"cls={cls_name}, rule={rule}, "
    #     f"all_cx={[round(item['cx'], 1) for item in instances]}, "
    #     f"selected_cx={selected['cx']:.1f}"
    # )

    return [selected]

def sq_dist_to(cx: float, cy: float):
    """Sort key: squared distance of an instance's centroid to (cx, cy)."""
    def _key(item) -> float:
        return (item["cx"] - cx) ** 2 + (item["cy"] - cy) ** 2
    return _key

def is_object_inside_basket(obj_mask, basket_cover_mask, ratio_thres=0.30):
    """
    Check whether an object instance is already inside the basket.

    overlap_ratio = overlap area of object and basket_cover_mask / object area.
    If overlap_ratio >= ratio_thres, the instance is considered inside the basket.
    """

    if obj_mask is None:
        return False

    obj_area = np.count_nonzero(obj_mask)
    if obj_area == 0:
        return False

    if basket_cover_mask is None or not np.any(basket_cover_mask):
        return False

    overlap_area = np.count_nonzero(obj_mask & basket_cover_mask)
    overlap_ratio = overlap_area / float(obj_area)

    return overlap_ratio >= ratio_thres

def select_basket_mask_by_side(labeled_baskets, side, h, w):
    """
    Select the basket mask to display / evaluate for the current task.

    side:
        "mid"   -> select mid_basket
        "right" -> select right_basket
        "all"   -> show all baskets
        None    -> show no basket, e.g. human hand mode
    """

    selected_mask = np.zeros((h, w), dtype=bool)

    if side is None:
        return selected_mask

    if len(labeled_baskets) == 0:
        return selected_mask

    # idle / initial state: show all baskets
    if side == "all":
        for item in labeled_baskets:
            if item["cls_name"] in ["mid_basket", "right_basket", "unknown_basket"]:
                selected_mask |= item["mask"]
        return selected_mask

    if side == "mid":
        target_label = "mid_basket"
    elif side == "right":
        target_label = "right_basket"
    else:
        return selected_mask

    for item in labeled_baskets:
        if item["cls_name"] == target_label:
            selected_mask |= item["mask"]

    return selected_mask

def select_basket_mask_by_sides(labeled_baskets, sides, h, w):
    """
    Get the basket mask for multiple basket sides.
    sides examples:
        {"mid"}
        {"right"}
        {"mid", "right"}
    """

    selected_mask = np.zeros((h, w), dtype=bool)

    if len(labeled_baskets) == 0:
        return selected_mask

    for side in sides:
        if side == "mid":
            target_label = "mid_basket"
        elif side == "right":
            target_label = "right_basket"
        else:
            continue

        for item in labeled_baskets:
            if item["cls_name"] == target_label:
                selected_mask |= item["mask"]

    return selected_mask

def make_basket_cover_mask(basket_union_mask):
    """
    Fill holes in the basket mask and estimate the basket interior with a convex hull.
    Returns a basket_cover_mask usable for overlap checks.
    """

    if basket_union_mask is None or not np.any(basket_union_mask):
        return None

    basket_cover_u8 = basket_union_mask.astype(np.uint8) * 255

    kernel = np.ones((9, 9), np.uint8)
    basket_cover_u8 = cv2.morphologyEx(
        basket_cover_u8,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        basket_cover_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_cover = np.zeros_like(basket_cover_u8)

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(
            filled_cover,
            [hull],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return filled_cover > 0
