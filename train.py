from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(model="yolov8x-seg.pt")
    model.train(data="Dataset/yolo_dataset/data.yaml",
                epochs=100,
                batch=16,
                imgsz=512,
                overlap_mask=True,
                save=True,
                save_period=5,
                device='mps',
                project='results',
                name='test',
                val=False,
                patience = 5)