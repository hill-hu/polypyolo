from multiprocessing import freeze_support

from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()
    # Load a model

    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='polyp_size_10mm.yaml', epochs=10, imgsz=640, batch=16)
