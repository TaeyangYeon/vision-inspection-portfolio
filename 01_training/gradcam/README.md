# GradCAM Custom Implementation

## Overview
Custom GradCAM implementation using PyTorch hooks.
No external library (pytorch-grad-cam) used.
This demonstrates deep understanding of model interpretability.

## How GradCAM Works
1. Forward pass: extract feature maps from target layer
2. Backward pass: compute gradients of class score w.r.t. feature maps
3. Global average pooling of gradients -> channel weights
4. Weighted sum of feature maps -> class activation map
5. ReLU + normalize -> heatmap

## Implementation Plan

### Step 1: Register hooks on target layer
- register_forward_hook -> save activation maps
- register_backward_hook -> save gradients

### Step 2: Forward + Backward pass
- Run inference on input image
- Select target class score
- Call .backward() to compute gradients

### Step 3: Compute CAM
- weights = mean(gradients, axis=[H, W])
- cam = sum(weights * activations, axis=channel)
- cam = relu(cam)
- cam = normalize to [0, 1]

### Step 4: Overlay on image
- Resize cam to original image size
- Apply colormap (cv2.COLORMAP_JET)
- Blend with original image

## Target Layer for YOLOv8
- model.model[-2] is the last C2f layer before detection head
- This captures high-level semantic features

## Files
- gradcam_core.py   : GradCAM class with PyTorch hooks
- gradcam_yolo.py   : YOLO-specific wrapper
- gradcam_test.py   : Validation against library result