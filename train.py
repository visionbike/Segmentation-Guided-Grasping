import albumentations as A
from ultralytics import YOLO

model_name = "yolo26"   # yolov11
model_size = "n"        # n, s
device_id = 2           # 0,1,2
num_epochs = 300
batch_size = 16
image_size = 640

# Ultralytics downloads a missing asset to whatever path it is given and creates
# the parent directory itself, so this keeps checkpoints out of the repo root.
pretrained_dir = "pretrained_models"

# Requires resplit_by_clip.py to have been run. The original dataset_yolo split is
# frame-level random, so all 16 video clips appear in train, val and test at once.
data_config = "dataset_yolo_split/data.yaml"

# Blur radii are in output pixels, i.e. relative to image_size. Roboflow's
# "0-2.5 px" was measured on its 320x320 preprocess canvas, so it doubles here.
blur_scale = image_size / 320

if __name__ == "__main__":
    model = YOLO(f"{pretrained_dir}/{model_name}{model_size}-seg.pt")   # fall back to yolo11n-seg.pt if unavailable
    model.train(
        data=data_config, epochs=num_epochs, imgsz=image_size, batch=batch_size, device=device_id,
        # segmentation-specific
        overlap_mask=True, mask_ratio=4,
        # geometry — degrees=180 subsumes Roboflow's 90-deg rots + its +/-15 deg;
        # flips add the reflections rotation alone cannot produce (no chiral classes here)
        degrees=180.0, translate=0.15, scale=0.4, shear=10.0, perspective=0.0005,
        fliplr=0.5, flipud=0.5,
        # photometric — hsv_v covers Roboflow's brightness + exposure; saturation
        # pulled back to 0.4 to protect the colour-coded snack/can classes
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        # blur robustness — non-spatial only; ultralytics' Albumentations hook
        # transforms bboxes but NOT segments (augment.py:2178), so anything
        # spatial here would silently corrupt the masks
        augmentations=[
            A.OneOf([
                # defocus / soft optics — matches PIL, so sigma == Roboflow radius
                A.GaussianBlur(blur_limit=0, sigma_limit=(0.25 * blur_scale, 2.0 * blur_scale), p=1.0),
                # wrist-cam motion during approach: random direction, non-uniform kernel
                A.MotionBlur(blur_limit=(5, 15), angle_range=(0, 360),
                             direction_range=(-1.0, 1.0), allow_shifted=True, p=1.0),
                # fixed-focus camera inside its minimum focus distance at grasp range
                A.Defocus(radius=(2, 7), alias_blur=(0.1, 0.3), p=1.0),
            ], p=0.35),   # OneOf, not a stack — composing all three destroys the image
        ],
        mosaic=1.0, close_mosaic=15, mixup=0.0,
        copy_paste=0.3, copy_paste_mode="flip",   # 20:1 imbalance, basket_bottle 86 vs robot_gripper 1879
        patience=50, seed=0,
        verbose=True,
    )
