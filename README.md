# AI-Powered Object Detection System

**CSO Project** — README and workflow

---

## Overview

This repository implements a workflow to build a filtered, custom YOLO-based object detection pipeline starting from COCO-like label sets. The codebase focuses on: filtering and fixing label/YAML files, creating a reduced dataset, preparing YOLO dataset structure, training a final YOLO model, and running evaluation/monitoring utilities.

Files in this repository (high-level):

* `create_filtered_dataset.py` / `create_filtered_dataset_clean.py` — build a smaller dataset from a larger COCO-style set, keeping only selected classes and copying images/labels to a training structure.
* `filtering.py` — helper functions to select classes, filter annotations and build intermediate lists.
* `fix_yaml.py` / `final_fix_yolo_structure.py` — correct and convert YAML files / rearrange directory structure to match YOLO training expectations.
* Several `coco_*.yaml` files — different YAML configurations for class sets and dataset splits.
* `debug_labels.py` — visualize or sanity-check labels (helpful for catching missing/incorrect boxes or class ids).
* `train_final_yolo.py` — script to kick off YOLO training on the prepared dataset (contains the training configuration and call to training library).
* `evaluate_yolo.py` — evaluate trained weights on validation/test set and produce metrics.
* `single_image.py` — run inference on a single image to test model outputs.
* `class_detection_counter.py` — analyze detection outputs across images (counts per-class detections, useful for imbalance analysis).
* `gpu_monitoring.py` / `performance_comp.py` — measure GPU usage, compare CPU vs GPU throughput, and produce simple performance comparisons.

(These filenames appear in the repository root.)

---

## Quick workflow (step-by-step)

Below is the recommended order of steps to reproduce the pipeline used in this project.

### 1) Set up environment

* Create a Python environment (recommended: `python 3.10` or later).
* Install dependencies: typical packages used for this pipeline are `numpy`, `opencv-python`, `pyyaml`, `tqdm`, and the YOLO training library you prefer (e.g., Ultralytics YOLO or YOLOv5). Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install numpy opencv-python pyyaml tqdm matplotlib seaborn
# plus the YOLO framework, e.g.:
# pip install ultralytics
```

### 2) Choose the classes to keep & YAML to use

* The repo contains several `coco_*.yaml` files. Pick one that best represents the desired subset of classes (e.g. `coco_filtered.yaml` or `coco_final.yaml`). These YAMLs map class names to indices and provide dataset paths expected by the training script.

### 3) Create the filtered dataset

* Use `create_filtered_dataset.py` or `create_filtered_dataset_clean.py` to generate a smaller dataset containing only selected classes. These scripts:

  * Read the original COCO-style annotations (JSON or label pairs),
  * Filter annotations to include only specified class IDs,
  * Copy/move images and labels into a standard YOLO directory structure — typically `images/train`, `images/val`, `labels/train`, `labels/val`.

Example:

```bash
python create_filtered_dataset_clean.py --input /path/to/original/annotations --out /path/to/output --yaml coco_filtered.yaml
```

(See script top-of-file `argparse` help for exact flags.)

### 4) Sanity-check labels

* Run `debug_labels.py` on a sample of the output dataset to visualize bounding boxes on images and confirm class IDs & box coordinates are correct.

```bash
python debug_labels.py --dataset /path/to/output --sample 50
```

### 5) Fix YAMLs & directory structure

* If you encounter mismatched class counts or YAML formatting problems, run `fix_yaml.py` to correct class lists and indices.
* Run `final_fix_yolo_structure.py` to ensure folders, train/val splits, and YAML references match what your YOLO trainer expects.

### 6) Train the final YOLO model

* Use `train_final_yolo.py` to launch training. The script references your chosen YAML (with paths and class names) and selects hyperparameters such as `epochs`, `batch_size`, and `img_size`.

Example (pseudo):

```bash
python train_final_yolo.py --data coco_final.yaml --epochs 50 --batch 16 --img 640
```

* Training outputs will typically be saved to a `runs/` or `weights/` folder — saved checkpoint `.pt` or `.pth` files are consumed by the evaluation and inference scripts.

### 7) Evaluate model performance

* Use `evaluate_yolo.py` with the trained weights and the validation dataset to obtain precision, recall, mAP, and class-wise breakdowns.

```bash
python evaluate_yolo.py --weights runs/exp/weights/best.pt --data coco_final.yaml
```

* `class_detection_counter.py` can be used on predictions to count detections per class and identify class imbalance or over/under detection.

### 8) Run inference (single-image / batch)

* Test using `single_image.py` for quick visual checks:

```bash
python single_image.py --weights runs/exp/weights/best.pt --img test.jpg --conf 0.25
```

* For bulk inference, modify `single_image.py` or use evaluation scripts to run over folders.

### 9) Monitor performance and compare hardware

* `gpu_monitoring.py` measures GPU utilization during training or inference (calls `nvidia-smi` or `pynvml` depending on implementation).
* `performance_comp.py` runs inference/training on CPU then GPU and reports FPS, memory usage, and wall-time comparisons. Use these outputs to choose optimal batch size and mixed precision settings.

---

## File-by-file purpose (short)

* `create_filtered_dataset*.py`: build reduced dataset copying only desired classes.
* `filtering.py`: filtering utilities (class lists, annotation trimming).
* `fix_yaml.py`: correct malformed dataset YAMLs.
* `final_fix_yolo_structure.py`: enforce final directory layout compatible with YOLO trainer.
* `debug_labels.py`: visual sanity checks for annotations.
* `train_final_yolo.py`: training entrypoint.
* `evaluate_yolo.py`: evaluation entrypoint.
* `single_image.py`: demo inference code.
* `class_detection_counter.py`: analytical tool for detection counts.
* `gpu_monitoring.py` / `performance_comp.py`: performance measurement.

---

## Recommended best-practices and tips

* Keep a deterministic split (seed your RNG) when creating train/val splits so results are reproducible.
* Validate a small sample with `debug_labels.py` before training — many training failures stem from bad label formatting or wrong class indices.
* Use mixed precision (if your trainer supports it) to speed up training and reduce GPU memory footprint.
* Track experiments (weights, config, metrics) in a structured `runs/` folder or use tools such as Weights & Biases.
* If you change class order in YAMLs, retrain from scratch — mismatched class indices will invalidate previous weights.

---

## Assumptions and notes

* This README is written from the files present (scripts and YAMLs) and typical YOLO workflows. Adjust command examples to the specific CLI options implemented in each script.
* Paths, flags, and exact CLI names may differ — inspect the top of each script for `argparse` help strings.

---

## Reproducibility checklist

* [ ] Python environment created and dependencies installed.
* [ ] Chosen YAML (`coco_final.yaml` or similar) updated to reflect intended class names and dataset paths.
* [ ] `create_filtered_dataset_clean.py` run and output validated with `debug_labels.py`.
* [ ] `final_fix_yolo_structure.py` run to ensure YOLO-compatible layout.
* [ ] Trained weights saved and evaluated with `evaluate_yolo.py`.

---


© CSO Project
