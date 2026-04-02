from ultralytics import YOLO
import cv2
import pandas as pd
from PIL import Image

def run_pcb_inference(image_pil, model_path="best.pt"):
    """
    Backend module to process the image, return annotations, and generate a data log.
    """
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(image_pil, conf=0.05, imgsz=640)
    
    # Process image overlay
    res_plotted = results[0].plot()
    annotated_img = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Process data logs for Milestone 4 export
    num_defects = len(results[0].boxes)
    defect_list = []
    found_classes = set()
    
    if num_defects > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            found_classes.add(cls_name)
            
            # Add each defect and its confidence score to our log
            defect_list.append({
                "Defect Type": cls_name, 
                "Confidence Score (%)": round(conf * 100, 2)
            })
            
    # Convert the list of defects into a Pandas DataFrame
    df = pd.DataFrame(defect_list) if defect_list else None
    
    return annotated_img, num_defects, list(found_classes), df