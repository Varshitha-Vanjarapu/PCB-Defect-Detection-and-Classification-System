# 🔍 PCB Defect Detection System
A deep learning project designed to automatically detect manufacturing defects on Printed Circuit Boards (PCBs) using computer vision.

## 📖 What This Project Does
This tool replaces manual human inspection. You can upload an image of a PCB, and the AI will instantly draw boxes around any microscopic errors.
It is trained to find 6 specific defects:
* Missing Hole
* Mouse Bite
* Open Circuit
* Short Circuit
* Spur
* Spurious Copper

## 📁 Project Folders & Milestones
The code is divided into three main folders to show how the project was built from start to finish:

### 1. Milestone_1_Data_Preprocessing
This folder contains the early baseline code. It includes scripts for image subtraction, Region of Interest (ROI) extraction and bounding box labeling to test basic computer vision techniques before using deep learning.

### 2. Milestone_2_Model_Training
This folder contains the code used to train the AI. It includes data annotation scripts, image augmentation, and the `train.py` and `test.py` files used to build the model.

### 3. Milestone_3_and_4_Frontend_Backend
This folder contains the final, working application. It includes the trained AI weights (`best.pt`) and a user-friendly Streamlit web dashboard.
