# Program 1: AI Model Trainer - Complete Description

## What is this program?

This is a training and evaluation tool for an artificial intelligence model
that automatically detects defects on product surfaces.

It runs in a web browser and allows you to train, evaluate, and export
an AI model without writing any code.

The purpose of this program is to simplify the development of AI-based
inspection systems so engineers can focus on improving model performance
rather than writing complex infrastructure code.

---

# Background Knowledge

## What is Artificial Intelligence (AI)?

Artificial Intelligence is a technology that allows computers to learn from
examples and make decisions on their own.

Just like a human inspector learns to spot defects by looking at thousands
of products, an AI model learns by looking at thousands of images.

Once trained, the AI model can analyze new images and detect defects
automatically.

---

## What is a Dataset?

A dataset is a collection of images used to teach the AI.

This project uses the **MVTec AD Dataset**, which is an industry-standard
benchmark dataset used by researchers and companies worldwide to test
defect detection systems.

Dataset information

- Total images: 5,354
- Categories: 15  
  (bottle, cable, carpet, grid, hazelnut, leather, metal nut, pill, screw, tile, toothbrush, transistor, wood, zipper, etc.)
- Each image is labeled with the type and location of defects

### Why the MVTec AD Dataset?

The MVTec AD dataset is widely used in industrial vision research because it
simulates real manufacturing defect scenarios.

Unlike typical image classification datasets, MVTec contains high-resolution
images of industrial products with carefully annotated defect regions.

Typical industrial challenges included in the dataset:

- subtle surface defects
- irregular defect shapes
- lighting variation
- very small anomaly regions

Using this dataset allows the model performance to be compared with other
research results in the field of industrial inspection.

---

## What is YOLO?

YOLO stands for **You Only Look Once**.

It is one of the most popular and fastest object detection algorithms
in the world.

Traditional detection methods scan an image multiple times,
but YOLO looks at the entire image only once and finds all objects
in a single pass.

This makes it extremely fast.

This project uses **YOLOv8**, the 8th generation of YOLO developed by Ultralytics.

YOLO is widely used in

- industrial inspection
- autonomous vehicles
- security systems
- robotics

Key facts about YOLOv8

- Can detect multiple objects in a single image
- Works in real time (30+ frames per second)
- Available in different sizes (n, s, m, l, x)
- This project uses **YOLOv8n (nano)** for fast inference on CPU

### Why YOLO for Industrial Inspection?

YOLO was selected for this project because industrial inspection systems
require **real-time performance**.

In a production line, products move continuously and the inspection system
must make decisions within milliseconds.

YOLO is well suited for this environment because:

- It processes the entire image in a single pass
- It provides real-time detection
- It balances speed and accuracy effectively
- It can run efficiently on CPU when exported to ONNX

These characteristics make YOLO widely used in manufacturing inspection
systems such as:

- bottle inspection
- PCB inspection
- automotive surface inspection
- semiconductor inspection

---

## What is ONNX?

ONNX stands for **Open Neural Network Exchange**.

When you train an AI model in Python using PyTorch, the model is saved
in a format that only Python can read.

ONNX converts the model into a **universal format** that any programming
language can use.

Think of it like converting a Word document (.docx) to PDF (.pdf)
so anyone can open it regardless of what software they have.

Benefits of ONNX

- Use the model in C#, Java, C++, or other languages
- Faster inference speed through graph optimization
- No need to install Python or PyTorch
- Easy deployment in production systems

This project exports the model to ONNX so that **Program 2 (C# inspection system)**
can run inference without Python.

---

## What is a Neural Network?

A neural network is a system inspired by the human brain.

It consists of layers of mathematical functions called **neurons** that are
connected to each other.

When an image is fed into the network:

1. Early layers detect edges
2. Middle layers detect shapes
3. Deep layers detect complex patterns

For example

- edges
- curves
- cracks
- contamination

These layers allow the model to understand complex visual patterns.

---

## What is Training?

Training is the process of teaching the AI model.

During training:

1. The model looks at a labeled image  
2. It makes a prediction  
3. The prediction is compared to the correct answer  
4. The error (loss) is calculated  
5. The model adjusts its internal parameters  

This process repeats thousands of times until the model becomes accurate.

This project trains on **Google Colab using a T4 GPU**
because training requires heavy computation.

---

## What is a GPU?

A GPU (Graphics Processing Unit) was originally designed for video games.

However, GPUs are extremely efficient at performing the mathematical
operations required for AI training.

A modern GPU can perform **thousands of operations simultaneously**.

This makes training **10–100x faster** than using a CPU.

This project uses the **Google Colab T4 GPU**.

---

## What is Streamlit?

Streamlit is a Python library that lets you build interactive web
applications using only Python.

Instead of building a frontend website separately,
Streamlit automatically creates:

- buttons
- charts
- sliders
- image viewers

This makes it ideal for **AI development tools**.

---

## What is GradCAM?

GradCAM stands for **Gradient-weighted Class Activation Mapping**.

It is a technique that visualizes which parts of an image the AI model
focused on when making a decision.

For example

If the model detects a crack in a bottle,
GradCAM generates a heatmap showing that the model focused on the crack.

Red / Yellow  
= important region

Blue  
= less important region

### Why Explainability Matters

In industrial inspection systems, it is important to verify that the AI model
is making decisions based on the **correct visual features**.

For example

If the model predicts a crack but focuses on the background,
the system may not be reliable in production.

GradCAM allows engineers to verify that the model focuses on
the actual defect area.

---

## What is Confusion Matrix?

A confusion matrix is a table showing how well the model performed.

Example

TP (True Positive)  
Correct detection

FP (False Positive)  
False alarm

FN (False Negative)  
Missed defect

TN (True Negative)  
Correct rejection

---

## What is mAP?

mAP stands for **mean Average Precision**.

It is the standard metric used to evaluate object detection models.

It combines

- precision
- recall

into a single score.

mAP50 means the score is calculated at **IoU threshold 0.5**.

This project achieved

**mAP50 = 0.869**

which means the model detects **86.9% of defects accurately**.

---

## What is a PR Curve?

A PR Curve (Precision-Recall Curve) shows the trade-off between

precision and recall.

Increasing threshold

→ higher precision  
→ lower recall

Lower threshold

→ higher recall  
→ more false alarms

---

## What is Data Augmentation?

Data augmentation increases dataset diversity by applying random
transformations to images.

Examples

- flip
- rotate
- brightness change
- noise

This helps the model generalize better.

---

# Project Overview

## System Pipeline

The inspection system follows a typical AI deployment pipeline.

1. Dataset Preparation  
   Collect and label defect images

2. Model Training  
   Train YOLOv8 on GPU

3. Model Evaluation  
   Evaluate using mAP, precision, recall

4. Model Export  
   Convert PyTorch model to ONNX

5. Deployment  
   Program 2 loads the ONNX model and runs real-time inspection

---

## Goal

Build a complete AI-powered surface defect inspection system.

The system consists of two programs.

Program 1  
AI model training and evaluation

Program 2  
Real-time inspection system

### Industrial Application

Surface defect detection is widely used in manufacturing.

Examples include

- glass bottle inspection
- semiconductor wafer inspection
- automotive parts inspection
- PCB inspection
- metal surface inspection

AI inspection systems improve consistency and reduce manual inspection work.

---

# Training Results

| Metric | Value |
|------|------|
| mAP50 | 0.869 |
| mAP50-95 | 0.677 |
| Precision | 0.807 |
| Recall | 0.800 |

---

# Inference Performance

| Metric | Value |
|------|------|
| Inference time | 36.2 ms |
| Estimated FPS | 27.6 |
| Model size | 12.3 MB |

Running inference on **CPU** is important for industrial environments
because many inspection machines do not have GPUs.

---

# Application Tabs

## Data Tab

Browse dataset images and preview augmentation effects.

---

## Train Tab

Configure training parameters

- model size
- epochs
- batch size
- learning rate

---

## Eval Tab

Evaluate trained model.

FP/FN analysis is very important in industrial inspection.

False Positive  
→ unnecessary product rejection

False Negative  
→ defective product passes inspection

The goal is to minimize FN while keeping FP acceptable.

---

## Export Tab

Export trained model to ONNX.

---

## GradCAM Tab

Visualize which regions the AI model focuses on.

---

# Key Technical Highlights

## Custom GradCAM Implementation

GradCAM implemented using raw PyTorch hooks.

---

## ONNX Runtime Inference

Model exported to ONNX for deployment in C# systems.

---

# Glossary

| Term | Meaning |
|----|----|
| YOLO | Object detection algorithm |
| ONNX | Universal AI model format |
| mAP | Detection accuracy metric |
| GradCAM | Model explainability heatmap |
| FP | False Positive |
| FN | False Negative |