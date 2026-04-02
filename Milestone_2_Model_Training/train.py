from ultralytics import YOLO
print("Initializing YOLOv8 Model...")
# Loads the YOLOv8 Nano model (fastest, great for internships and laptops)
model = YOLO('yolov8n.pt') 

print("Starting Training...")
# Train the model. It automatically reads your data.yaml!
results = model.train(
    data='data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16,
    plots=True # Generates beautiful graphs for your mentor presentation
)

print("Training Complete! Your weights are saved in runs/detect/train/weights/best.pt")