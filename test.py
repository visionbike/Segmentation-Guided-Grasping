from ultralytics import YOLO

image_size = 640
data_config = "./dataset_yolo_split/data.yaml"

# Must be a run trained on dataset_yolo_split. Weights from the old dataset_yolo
# runs (train..train-4) saw these test images during training, because that split
# was frame-level random -- their scores here are meaningless.
weights = "./runs/segment/train-5/weights/best.pt"

if __name__ == "__main__":
    model = YOLO(weights)
    results = model.val(data=data_config, split="test", imgsz=image_size, verbose=True)
    for c in results.box.ap_class_index:
        print(
            f"{model.names[c]:<16} boxAP50-95 {results.box.maps[c]:.3f} maskAP50-95 {results.seg.maps[c]:.3f}"
        )
    print("pooled mask mAP50-95:", results.seg.map)

    model.predict("dataset_yolo_split/images/test", save=True, conf=0.1, project="runs/segment", name="test_pred")
