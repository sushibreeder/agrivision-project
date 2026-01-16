from ultralytics import YOLO

if __name__ == '__main__':
    # Load the model
    model = YOLO('yolov8n.pt') 

    # Train the model
    results = model.train(
        data='datasets/CropWeeds/data.yaml', 
        epochs=3,          # Short run for demo
        imgsz=640,
        device='cpu'       # Use CPU for laptop
    )