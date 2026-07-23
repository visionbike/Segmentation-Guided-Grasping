from ultralytics.data.converter import convert_coco

if __name__ == "__main__":
    convert_coco("coco_json", save_dir="dataset_yolo26", use_segments=True, cls91to80=False)

