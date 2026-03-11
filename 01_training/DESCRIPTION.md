Content:
---
# Program 1: AI Model Trainer - Complete Description

## What is this program?

This is a training and evaluation tool for an artificial intelligence model
that automatically detects defects on product surfaces.
It runs in a web browser and allows you to train, evaluate, and export
an AI model without writing any code.

---

## Background Knowledge

### What is Artificial Intelligence (AI)?

Artificial Intelligence is a technology that allows computers to learn from
examples and make decisions on their own.

Just like a human inspector learns to spot defects by looking at thousands
of products, an AI model learns by looking at thousands of images.

### What is a Dataset?

A dataset is a collection of images used to teach the AI.

This project uses the MVTec AD Dataset, which is an industry-standard
benchmark dataset used by researchers and companies worldwide to test
defect detection systems.

- Total images: 5,354
- Categories: 15 (bottle, cable, carpet, grid, hazelnut, leather, metal nut,
  pill, screw, tile, toothbrush, transistor, wood, zipper, and more)
- Each image is labeled with the type and location of defects

### What is YOLO?

YOLO stands for "You Only Look Once."

It is one of the most popular and fastest object detection algorithms in
the world. Traditional detection methods scan an image multiple times,
but YOLO looks at the entire image only once and finds all objects in
a single pass. This makes it extremely fast.

This project uses YOLOv8, which is the 8th generation of YOLO developed
by Ultralytics. It is widely used in industrial inspection, autonomous
vehicles, and security systems.

Key facts about YOLOv8:
- Can detect multiple objects in a single image
- Works in real-time (30+ frames per second)
- Available in different sizes: nano (n), small (s), medium (m), large (l), xlarge (x)
- This project uses YOLOv8n (nano) for fast inference on CPU

### What is ONNX?

ONNX stands for "Open Neural Network Exchange."

When you train an AI model in Python using PyTorch, the model is saved
in a format that only Python can read. ONNX converts the model into a
universal format that any programming language can use.

Think of it like converting a Word document (.docx) to PDF (.pdf) so
anyone can open it regardless of what software they have.

Benefits of ONNX:
- Use the model in C#, Java, C++, or any other language
- Faster inference speed through graph optimization
- No need to install Python or PyTorch on the deployment machine
- This project exports to ONNX so that Program 2 (C#) can run inference
  without Python

### What is a Neural Network?

A neural network is a system inspired by the human brain.

It consists of layers of mathematical functions called "neurons" that
are connected to each other. When an image is fed into the network,
it passes through these layers and each layer extracts different
features - edges in the first layer, shapes in the middle layers,
and complex patterns like "broken bottle" in the final layers.

### What is Training?

Training is the process of teaching the AI model.

During training:
1. The model looks at a labeled image (e.g., a bottle with a crack)
2. It makes a prediction (e.g., "I think there is a scratch here")
3. The prediction is compared to the correct answer
4. The difference (called "loss") is calculated
5. The model adjusts its internal parameters to reduce the loss
6. This process repeats thousands of times until the model is accurate

This project trains on Google Colab using a T4 GPU because training
requires heavy computation that a regular laptop cannot handle efficiently.

### What is a GPU?

A GPU (Graphics Processing Unit) was originally designed for video games
to render graphics quickly. Researchers discovered that GPUs are also
extremely efficient at the mathematical operations required for AI training.

A modern GPU can perform thousands of operations simultaneously, making
training 10-100x faster than using a regular CPU.

This project uses Google Colab's free T4 GPU for training.

### What is Streamlit?

Streamlit is a Python library that lets you build interactive web
applications with pure Python code.

Instead of building a separate frontend website, you write Python code
and Streamlit automatically creates buttons, sliders, charts, and image
viewers. This makes it ideal for data science and AI tools.

### What is GradCAM?

GradCAM stands for "Gradient-weighted Class Activation Mapping."

It is a technique that visualizes which parts of an image the AI model
focused on when making a decision.

For example, if the model detects a crack in a bottle, GradCAM generates
a heatmap showing that the model was looking at the crack area (shown in
red/yellow) and not at the background (shown in blue).

This is important for:
- Understanding why the model made a certain decision
- Verifying that the model is looking at the right features
- Debugging false detections

In this project, GradCAM is implemented from scratch using pure PyTorch
hooks (register_forward_hook and register_full_backward_hook) without
using any external library. This demonstrates deep understanding of how
neural networks work internally.

### What is Confusion Matrix?

A confusion matrix is a table that shows how well the model performed
on each defect class.

For example:
- True Positive (TP): Model said "crack" and it was actually a crack
- False Positive (FP): Model said "crack" but there was no crack (false alarm)
- False Negative (FN): Model said "no defect" but there was actually a crack (missed defect)
- True Negative (TN): Model said "no defect" and it was actually clean

A good model has high TP and low FP and FN.

### What is mAP?

mAP stands for "mean Average Precision."

It is the standard metric used to evaluate object detection models.
It combines both precision (how accurate the detections are) and
recall (how many defects were found) into a single score.

mAP50 means the score is calculated at IoU threshold 0.5.
IoU (Intersection over Union) measures how well the predicted bounding
box matches the actual defect location.

This project achieved mAP50 = 0.869 on the bottle category, which means
the model correctly detects 86.9% of defects with accurate bounding boxes.

### What is a PR Curve?

A PR Curve (Precision-Recall Curve) shows the trade-off between precision
and recall at different confidence thresholds.

Moving the confidence threshold higher makes the model more precise but
it will miss more defects. Moving it lower will catch more defects but
with more false alarms. The curve helps you choose the right threshold
for your specific use case.

### What is Data Augmentation?

Data augmentation is a technique to artificially increase the size of
the training dataset by applying random transformations to existing images.

For example, a single image of a broken bottle can be transformed into
many different versions by flipping it horizontally, rotating it,
changing brightness, adding noise, and so on.

This helps the model become more robust and generalize better to
real-world conditions where lighting, angle, and position may vary.

---

## Project Overview

### Goal

Build a complete AI-powered surface defect inspection system consisting
of two programs:

- Program 1 (this program): Train and evaluate the AI model
- Program 2: Real-time inspection system using the trained model

### Dataset Used

MVTec AD (Anomaly Detection) Dataset

This project focuses on the "bottle" category which has 3 defect classes:
- broken_large: Large cracks or breaks on the bottle surface
- broken_small: Small cracks or chips on the bottle surface
- contamination: Foreign substances or dirt on the bottle surface

### Training Environment

- Platform: Google Colab (free cloud GPU service by Google)
- GPU: NVIDIA T4 (16GB VRAM)
- Training time: approximately 1 hour for 100 epochs
- Framework: PyTorch + Ultralytics YOLOv8

### Training Results

| Metric        | Value  |
|---------------|--------|
| mAP50         | 0.869  |
| mAP50-95      | 0.677  |
| Precision     | 0.807  |
| Recall        | 0.800  |
| broken_large  | 0.912  |
| broken_small  | 0.898  |
| contamination | 0.798  |

### Inference Performance

| Metric          | Value         |
|-----------------|---------------|
| Inference time  | 36.2 ms/frame |
| Estimated FPS   | 27.6          |
| Model size      | 12.3 MB       |
| Runtime         | ONNX CPU      |
| Hardware        | Intel Mac     |

---

## How to Run

### Requirements

- Python 3.11
- pyenv (Python version manager)
- All dependencies listed in requirements.txt

### Setup
```bash
cd 01_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the Application
```bash
cd 01_training
source .venv/bin/activate
streamlit run app/main.py
```

Then open your browser at: http://localhost:8501

---

## Application Tabs

### Data Tab

Browse the dataset images with bounding box overlays.
Select category (bottle), split (train/val), and image.
View class distribution charts.
Preview data augmentation effects with interactive sliders.

### Train Tab

Configure training parameters:
- Model size (nano/small/medium/large/xlarge)
- Number of epochs
- Batch size
- Learning rate
- Augmentation settings

View training results with interactive charts:
- Loss curves (box loss, class loss, DFL loss)
- mAP progression over epochs

### Eval Tab

Evaluate the trained model with:
- Interactive Confusion Matrix (Plotly heatmap)
- Per-class metrics table (AP50, Precision, Recall, F1)
- PR Curve visualization
- F1 Score bar chart
- Sample image inference viewer (runs real ONNX model)
- FP/FN case analysis with color-coded bounding boxes:
  Green = True Positive (correct detection)
  Red = False Positive (false alarm)
  Blue = False Negative (missed defect)

### Export Tab

- View model file information (ONNX size, PT size)
- Run inference speed benchmark
- Visualize ONNX detection results on sample images

### GradCAM Tab

- Visualize which regions the model focuses on
- Single class mode: generate heatmap for one defect class
- All classes mode: compare heatmaps across all defect classes side by side
- View activation statistics (min, max, mean)

---

## File Structure

01_training/
├── app/
│   ├── main.py                    # Streamlit app entry point
│   ├── pages/
│   │   ├── data_tab.py            # Data browsing and augmentation
│   │   ├── train_tab.py           # Training configuration and results
│   │   ├── eval_tab.py            # Evaluation metrics and FP/FN analysis
│   │   ├── export_tab.py          # ONNX export and speed benchmark
│   │   └── gradcam_tab.py         # GradCAM heatmap visualization
│   ├── components/
│   │   └── augmentation.py        # Augmentation preview component
│   └── utils/
│       ├── model_utils.py         # ONNX inference utilities
│       ├── fpfn_utils.py          # FP/FN classification utilities
│       ├── data_loader.py         # Dataset loading utilities
│       ├── health_check.py        # System health check
│       └── integration_test.py    # Integration test runner
├── gradcam/
│   ├── gradcam_core.py            # Custom GradCAM with PyTorch hooks
│   ├── gradcam_yolo.py            # YOLO-specific GradCAM wrapper
│   ├── gradcam_visualize.py       # Layer-by-layer activation visualizer
│   ├── gradcam_multiclass.py      # Multi-class comparison generator
│   └── gradcam_test.py            # GradCAM test script
├── data/
│   ├── raw/                       # Original MVTec AD dataset
│   └── processed/                 # Converted to YOLO format
├── models/
│   └── best.onnx                  # Trained and exported ONNX model (12.3 MB)
├── outputs/
│   └── bottle_full/
│       ├── results.csv            # Training metrics per epoch
│       ├── eval_results.json      # Evaluation results (mAP, confusion matrix)
│       └── pr_data.json           # PR curve data
└── scripts/
├── explore_data.py            # Dataset exploration script
├── convert_to_yolo.py         # MVTec mask to YOLO bbox converter
└── visualize_labels.py        # Label visualization script

---

## Key Technical Highlights

### Custom GradCAM Implementation

GradCAM is implemented using raw PyTorch hooks without any external
library. This is more challenging but demonstrates deeper understanding:
```python
# Register forward hook to capture activations
def forward_hook(module, input, output):
    self.activations.append(output.detach())

# Register backward hook to capture gradients
def backward_hook(module, grad_input, grad_output):
    self.gradients.append(grad_output[0].detach())

self.target_layer.register_forward_hook(forward_hook)
self.target_layer.register_full_backward_hook(backward_hook)
```

### MVTec Mask to YOLO Conversion

The MVTec dataset provides defect regions as PNG masks (white pixels
on black background). These must be converted to YOLO format bounding
boxes using contour detection:
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Convert to YOLO normalized format: cx cy w h
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
```

### ONNX Runtime Inference (No Python Required)

The trained model is exported to ONNX format so that Program 2 (C#)
can run inference without installing Python or PyTorch.
The ONNX model takes a 640x640 float32 tensor as input and outputs
bounding box predictions that are then filtered using NMS
(Non-Maximum Suppression).

---

## Glossary

| Term              | Meaning                                                      |
|-------------------|--------------------------------------------------------------|
| AI / ML           | Artificial Intelligence / Machine Learning                   |
| YOLO              | You Only Look Once - fast object detection algorithm         |
| YOLOv8            | 8th generation YOLO by Ultralytics                           |
| ONNX              | Open Neural Network Exchange - universal model format        |
| PyTorch           | Python deep learning framework by Meta                       |
| Ultralytics       | Company that created YOLOv8                                  |
| MVTec AD          | Industry-standard defect detection benchmark dataset         |
| mAP50             | Mean Average Precision at IoU 0.5 threshold                  |
| IoU               | Intersection over Union - measures bounding box accuracy     |
| Epoch             | One complete pass through the training dataset               |
| Batch size        | Number of images processed together in one training step     |
| Learning rate     | How fast the model adjusts its parameters during training    |
| Augmentation      | Artificial image transformations to increase dataset size    |
| GradCAM           | Heatmap showing which image regions the model focused on     |
| Confusion Matrix  | Table showing correct and incorrect predictions per class    |
| PR Curve          | Precision-Recall trade-off curve at different thresholds     |
| FP                | False Positive - model detected a defect that does not exist |
| FN                | False Negative - model missed an actual defect               |
| TP                | True Positive - model correctly detected a real defect       |
| NMS               | Non-Maximum Suppression - removes duplicate bounding boxes   |
| Streamlit         | Python library for building interactive web applications     |
| Tensor            | Multi-dimensional array used to represent images in AI       |
| Inference         | Using a trained model to make predictions on new images      |
| GPU               | Graphics Processing Unit - accelerates AI training           |
| Colab             | Google Colab - free cloud-based GPU computing platform       |
---
