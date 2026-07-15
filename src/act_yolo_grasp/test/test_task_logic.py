# -*- coding: utf-8 -*-
"""Unit tests for act_yolo_grasp.task_logic (pure, ROS-free).

Run from the package root with either:
    pytest test/test_task_logic.py
    colcon test --packages-select act_yolo_grasp
"""

import numpy as np

from act_yolo_grasp import task_logic


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
H, W = 240, 320


def square_mask(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


BASKET_RULE = {
    "top": {"left_label": "mid_basket", "right_label": "right_basket"},
    "front": {"left_label": "right_basket", "right_label": "mid_basket"},
}
BASKET_SINGLE = {
    "top": "unknown_basket",
    "left_wrist": "mid_basket",
}
OBJECT_RULE = {
    "top": {"left_label": "right_object", "right_label": "mid_object"},
}
OBJECT_SINGLE = {
    "top": "unknown_object",
}
SELECT_RULE = {
    "top": "max_x",
    "front": "min_x",
}


# ------------------------------------------------------------------
# is_object_inside_basket
# ------------------------------------------------------------------
class TestIsObjectInsideBasket:
    def test_fully_inside(self):
        obj = square_mask(100, 120, 100, 120)
        cover = square_mask(90, 130, 90, 130)
        assert task_logic.is_object_inside_basket(obj, cover, 0.30)

    def test_fully_outside(self):
        obj = square_mask(0, 20, 0, 20)
        cover = square_mask(90, 130, 90, 130)
        assert not task_logic.is_object_inside_basket(obj, cover, 0.30)

    def test_partial_overlap_threshold(self):
        # object 20x20 = 400 px; overlap region 10x20 = 200 px -> ratio 0.5
        obj = square_mask(100, 120, 100, 120)
        cover = square_mask(100, 120, 110, 200)
        assert task_logic.is_object_inside_basket(obj, cover, ratio_thres=0.5)
        assert not task_logic.is_object_inside_basket(obj, cover, ratio_thres=0.51)

    def test_none_or_empty_inputs(self):
        obj = square_mask(100, 120, 100, 120)
        empty = np.zeros((H, W), dtype=bool)
        assert not task_logic.is_object_inside_basket(None, obj, 0.30)
        assert not task_logic.is_object_inside_basket(empty, obj, 0.30)
        assert not task_logic.is_object_inside_basket(obj, None, 0.30)
        assert not task_logic.is_object_inside_basket(obj, empty, 0.30)


# ------------------------------------------------------------------
# relabel_basket_instances
# ------------------------------------------------------------------
class TestRelabelBasketInstances:
    def test_empty(self):
        assert task_logic.relabel_basket_instances([], "top", BASKET_RULE, BASKET_SINGLE) == []

    def test_single_basket_uses_single_label_map(self):
        inst = [{"mask": square_mask(0, 10, 0, 10), "cx": 50.0}]
        out = task_logic.relabel_basket_instances(inst, "top", BASKET_RULE, BASKET_SINGLE)
        assert out[0]["cls_name"] == "unknown_basket"

        out = task_logic.relabel_basket_instances(inst, "left_wrist", BASKET_RULE, BASKET_SINGLE)
        assert out[0]["cls_name"] == "mid_basket"

    def test_two_baskets_labeled_by_x(self):
        m = square_mask(0, 10, 0, 10)
        inst = [{"mask": m, "cx": 250.0}, {"mask": m, "cx": 50.0}]  # deliberately unsorted
        out = task_logic.relabel_basket_instances(inst, "top", BASKET_RULE, BASKET_SINGLE)
        assert [o["cls_name"] for o in out] == ["mid_basket", "right_basket"]
        assert out[0]["cx"] == 50.0 and out[1]["cx"] == 250.0

    def test_front_camera_rule_is_reversed(self):
        m = square_mask(0, 10, 0, 10)
        inst = [{"mask": m, "cx": 50.0}, {"mask": m, "cx": 250.0}]
        out = task_logic.relabel_basket_instances(inst, "front", BASKET_RULE, BASKET_SINGLE)
        assert [o["cls_name"] for o in out] == ["right_basket", "mid_basket"]

    def test_three_baskets_keeps_only_min_and_max_x(self):
        m = square_mask(0, 10, 0, 10)
        inst = [{"mask": m, "cx": 50.0}, {"mask": m, "cx": 150.0}, {"mask": m, "cx": 250.0}]
        out = task_logic.relabel_basket_instances(inst, "top", BASKET_RULE, BASKET_SINGLE)
        assert len(out) == 2
        assert {o["cx"] for o in out} == {50.0, 250.0}


# ------------------------------------------------------------------
# relabel_object_instances
# ------------------------------------------------------------------
class TestRelabelObjectInstances:
    def test_empty(self):
        assert task_logic.relabel_object_instances([], "top", OBJECT_RULE, OBJECT_SINGLE) == []

    def test_single_object_gets_unknown_label(self):
        inst = [{"cls_name": "tea", "mask": square_mask(0, 10, 0, 10), "cx": 50.0, "cy": 10.0}]
        out = task_logic.relabel_object_instances(inst, "top", OBJECT_RULE, OBJECT_SINGLE)
        assert out[0]["position_label"] == "unknown_object"
        assert out[0]["cls_name"] == "tea"

    def test_two_objects_rule_is_reverse_of_basket(self):
        m = square_mask(0, 10, 0, 10)
        inst = [{"cls_name": "tea", "mask": m, "cx": 50.0, "cy": 10.0},
                {"cls_name": "tea", "mask": m, "cx": 250.0, "cy": 10.0}]
        out = task_logic.relabel_object_instances(inst, "top", OBJECT_RULE, OBJECT_SINGLE)
        assert [o["position_label"] for o in out] == ["right_object", "mid_object"]


# ------------------------------------------------------------------
# select_object_instance_by_camera_x
# ------------------------------------------------------------------
class TestSelectObjectInstanceByCameraX:
    INSTANCES = [{"cls_name": "tea", "cx": 50.0, "cy": 10.0},
                 {"cls_name": "tea", "cx": 250.0, "cy": 10.0}]

    def test_empty(self):
        assert task_logic.select_object_instance_by_camera_x([], "top", SELECT_RULE) == []

    def test_single_passthrough(self):
        out = task_logic.select_object_instance_by_camera_x(self.INSTANCES[:1], "top", SELECT_RULE)
        assert out == [self.INSTANCES[0]]

    def test_max_x_rule(self):
        out = task_logic.select_object_instance_by_camera_x(self.INSTANCES, "top", SELECT_RULE)
        assert out[0]["cx"] == 250.0

    def test_min_x_rule(self):
        out = task_logic.select_object_instance_by_camera_x(self.INSTANCES, "front", SELECT_RULE)
        assert out[0]["cx"] == 50.0

    def test_unknown_camera_defaults_to_max_x(self):
        out = task_logic.select_object_instance_by_camera_x(self.INSTANCES, "nonexistent", SELECT_RULE)
        assert out[0]["cx"] == 250.0


# ------------------------------------------------------------------
# make_basket_cover_mask
# ------------------------------------------------------------------
class TestMakeBasketCoverMask:
    def test_none_and_empty_return_none(self):
        assert task_logic.make_basket_cover_mask(None) is None
        assert task_logic.make_basket_cover_mask(np.zeros((H, W), dtype=bool)) is None

    def test_solid_square_covered(self):
        m = square_mask(100, 140, 100, 140)
        cover = task_logic.make_basket_cover_mask(m)
        assert cover is not None
        assert cover[m].all()

    def test_hollow_basket_interior_is_filled(self):
        # U-shaped rim (open top): the convex hull must cover the interior
        rim = square_mask(100, 140, 100, 140) & ~square_mask(100, 135, 105, 135)
        cover = task_logic.make_basket_cover_mask(rim)
        assert cover is not None
        interior = square_mask(110, 130, 110, 130)
        assert cover[interior].all()


# ------------------------------------------------------------------
# select_basket_mask_by_side / _by_sides
# ------------------------------------------------------------------
class TestSelectBasketMask:
    MID = square_mask(100, 120, 40, 60)
    RIGHT = square_mask(100, 120, 260, 280)
    UNKNOWN = square_mask(0, 20, 0, 20)
    LABELED = [
        {"cls_name": "mid_basket", "mask": MID, "cx": 50.0},
        {"cls_name": "right_basket", "mask": RIGHT, "cx": 270.0},
        {"cls_name": "unknown_basket", "mask": UNKNOWN, "cx": 10.0},
    ]

    def test_mid_side(self):
        m = task_logic.select_basket_mask_by_side(self.LABELED, "mid", H, W)
        assert m.sum() == self.MID.sum() and m[self.MID].all()

    def test_right_side(self):
        m = task_logic.select_basket_mask_by_side(self.LABELED, "right", H, W)
        assert m[self.RIGHT].all() and not m[self.MID].any()

    def test_none_side_shows_nothing(self):
        assert task_logic.select_basket_mask_by_side(self.LABELED, None, H, W).sum() == 0

    def test_all_side_includes_unknown(self):
        m = task_logic.select_basket_mask_by_side(self.LABELED, "all", H, W)
        assert m[self.MID].all() and m[self.RIGHT].all() and m[self.UNKNOWN].all()

    def test_empty_instances(self):
        assert task_logic.select_basket_mask_by_side([], "mid", H, W).sum() == 0

    def test_by_sides_union(self):
        m = task_logic.select_basket_mask_by_sides(self.LABELED, {"mid", "right"}, H, W)
        assert m[self.MID].all() and m[self.RIGHT].all() and not m[self.UNKNOWN].any()

    def test_by_sides_empty_set(self):
        assert task_logic.select_basket_mask_by_sides(self.LABELED, set(), H, W).sum() == 0


# ------------------------------------------------------------------
# sq_dist_to
# ------------------------------------------------------------------
class TestSqDistTo:
    def test_exact_value(self):
        assert task_logic.sq_dist_to(10, 10)({"cx": 13, "cy": 14}) == 25.0

    def test_min_selection_tracks_nearest(self):
        instances = [{"cx": 10.0, "cy": 10.0}, {"cx": 50.0, "cy": 50.0}, {"cx": 12.0, "cy": 9.0}]
        nearest = min(instances, key=task_logic.sq_dist_to(11.0, 10.0))
        assert nearest == {"cx": 10.0, "cy": 10.0}
