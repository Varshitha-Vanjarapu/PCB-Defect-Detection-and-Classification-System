from ultralytics import YOLO

def run_evaluation(model_weights="best.pt", dataset_yaml="data.yaml"):
    """
    Evaluates the trained YOLOv8 model on the hold-out test dataset.
    """
    print(f"Loading YOLOv8 weights from {model_weights}...")
    model = YOLO(model_weights)
    
    print("Initiating testing sequence...")
    # 'split=test' ensures it evaluates purely on the test set defined in your yaml
    metrics = model.val(data=dataset_yaml, split='test')
    
    print("\n---Evaluation Complete ---")
    # metrics.box contains the evaluation results for object detection
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
    print(f"Mean Average Precision (mAP50):    {metrics.box.map50:.4f}")
    print(f"Overall Precision:                 {metrics.box.mp:.4f}")
    print(f"Overall Recall:                    {metrics.box.mr:.4f}")

if __name__ == "__main__":
    run_evaluation()