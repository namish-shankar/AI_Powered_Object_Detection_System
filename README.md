# AI-Powered Object Detection System

**CSO Project** — README and workflow

---

## Overview

This repository implements a workflow for building a filtered, custom YOLO-based object-detection pipeline from COCO-style label sets. It covers: filtering & fixing label/YAML files, creating a reduced dataset, preparing a YOLO-compatible dataset structure, training a final YOLO model, evaluating results, and monitoring performance.

---

## Workflow (step-by-step)

### 1) Set up environment  
Create and activate a Python environment with required dependencies (e.g., `numpy`, `opencv-python`, `pyyaml`, `tqdm`, etc.), plus the YOLO framework you’ll use.

### 2) Choose classes & YAML  
Select a YAML configuration file that defines the subset of classes you intend to keep, and that sets dataset paths expected by your training script.

### 3) Create the filtered dataset  
Use the filtering scripts to generate a smaller dataset containing only the selected classes. The output dataset will follow the standard YOLO directory structure with images and labels (train/val).

### 4) Sanity-check labels  
Visualize a sample of the output dataset to ensure bounding boxes, class IDs, and annotation formats are correct.

### 5) Fix YAML & directory structure  
If issues arise (e.g., class-count mismatches, folder layout problems), use the YAML-fixing and structure-fixing utilities to make your dataset compatible with the YOLO training trader.

### 6) Train the final YOLO model  
Launch training on the prepared dataset using your chosen YAML config, selecting appropriate hyperparameters (epochs, batch size, img size).  
Store the resulting model weights for evaluation and inference.

### 7) Evaluate model performance  
Use an evaluation script to compute key metrics — precision, recall, mAP — on validation/test sets. Analyze per-class detection performance and imbalance if required.

### 8) Run inference  
Test the trained model on individual images or batches to visually confirm detection outputs and overall behavior.

### 9) Monitor performance & hardware usage  
Use monitoring utilities to check GPU (or CPU) utilization during training/inference. Compare hardware performance (FPS, memory usage, wall time) and adjust settings (e.g., mixed precision) accordingly.

---

## File-by-file purpose

| File | Purpose |
|------|---------|
| `create_filtered_dataset_clean.py` | Build a reduced dataset by selecting desired classes and copying images/labels into a YOLO-compatible structure. |
| `filtering.py` | Provide utility functions for class filtering, annotation trimming and intermediate lists creation. |
| `fix_yaml.py` | Correct or update YAML dataset configurations (class lists, indices, paths) to match requirements. |
| `final_fix_yolo_structure.py` | Adjust directory layout, train/val folder split and YAML references so they align with YOLO trainer expectations. |
| `debug_labels.py` | Visual inspection tool to overlay bounding boxes on images and check class IDs / coordinate correctness. |
| `train_final_yolo.py` | Entry point for training a YOLO model on the prepared dataset using your configuration. |
| `evaluate_yolo.py` | Evaluate trained model weights on validation/test set and generate metrics (mAP, precision/recall, class-breakdown). |
| `single_image.py` | Script to run model inference on a single image for quick visual validation of detection outputs. |
| `class_detection_counter.py` | Analyze detection outputs (e.g., count detections per class) to understand class-wise performance / imbalance issues. |
| `gpu_monitoring.py` | Monitor GPU (or CPU) utilization, memory usage during training or inference. |
| `performance_comp.py` | Compare hardware performance (CPU vs GPU) for training/inference, measuring FPS, memory usage and wall-time. |

---

© CSO Project
