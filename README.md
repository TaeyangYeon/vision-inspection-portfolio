# Vision Inspection Portfolio

A complete computer vision pipeline demonstrating YOLOv8 fine-tuning for defect detection and real-time inspection system deployment for semiconductor and manufacturing environments.

## Architecture Diagram

```
┌──────────────────────────────────┐    ┌─────────────────────────────────┐
│        Program 1: Trainer        │    │    Program 2: Inspection        │
│       (Python + Streamlit)       │    │      (C# .NET + Avalonia)       │
├──────────────────────────────────┤    ├─────────────────────────────────┤
│                                  │    │                                 │
│  ┌─────────────────────────────┐ │    │ ┌─────────────────────────────┐ │
│  │     MVTec AD Dataset        │ │    │ │      ONNX Runtime           │ │
│  │    15 categories, 5354      │ │    │ │    CPU Inference Engine     │ │
│  └─────────────────────────────┘ │    │ └─────────────────────────────┘ │
│               │                  │    │               │                 │
│  ┌─────────────────────────────┐ │    │ ┌─────────────────────────────┐ │
│  │      YOLOv8 Training        │ │────┤ │      Real-time GUI          │ │
│  │    PyTorch + Ultralytics    │ │    │ │   Inspection / Benchmark    │ │
│  └─────────────────────────────┘ │    │ └─────────────────────────────┘ │
│               │                  │    │               │                 │
│  ┌─────────────────────────────┐ │    │ ┌─────────────────────────────┐ │
│  │    Custom GradCAM           │ │    │ │      Test Suite             │ │
│  │   PyTorch Hooks Analysis    │ │    │ │    150 NUnit Tests          │ │
│  └─────────────────────────────┘ │    │ └─────────────────────────────┘ │
│               │                  │    │                                 │
└───────────────┼──────────────────┘    └─────────────────────────────────┘
                │
        ┌───────────────┐
        │  ONNX Export  │
        │  .onnx model  │
        └───────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Training | Python 3.11, PyTorch, YOLOv8, Ultralytics |
| Inference | C# .NET 10, ONNX Runtime |
| UI Framework | Streamlit (Python), Avalonia UI (C#) |
| Testing | NUnit, Moq, coverlet (C#) |
| Dataset | MVTec Anomaly Detection (15 categories, 5354 images) |

## Program 1 - Model Trainer

A comprehensive YOLOv8 training pipeline with custom visualization tools for defect detection model development. The system provides data preprocessing, model training with hyperparameter tuning, evaluation metrics, ONNX export capabilities, and custom GradCAM implementation for explainable AI analysis. Built with Streamlit for intuitive web interface across five main tabs covering the complete ML workflow.

### Training Results

| Category | Model   | mAP50 | mAP50-95 | Precision | Recall |
|----------|---------|-------|----------|-----------|--------|
| bottle   | YOLOv8n | 0.869 | 0.677    | 0.807     | 0.800  |
| tile     | YOLOv8n | 0.946 | 0.720    | 0.938     | 0.855  |

**Note**: GradCAM implementation uses custom PyTorch forward and backward hooks without external visualization libraries, providing direct feature map analysis for model interpretability.

## Program 2 - Inspection System

A production-ready real-time inspection system built with C# .NET 10 and Avalonia UI for cross-platform deployment in manufacturing environments. The system loads ONNX models for CPU-based inference, provides comprehensive benchmarking tools, and includes session logging with CSV export functionality. Architecture follows SOLID principles with MVVM pattern and comprehensive test coverage.

### Inference Performance

| Model    | Description           | Avg (ms) | Min (ms) | Max (ms) | FPS  |
|----------|-----------------------|----------|----------|----------|------|
| bottle_n | YOLOv8n Bottle Detection | 35.74    | 27.68    | 49.35    | 27.98 |
| bottle_s | YOLOv8s Bottle Detection | 88.79    | 74.40    | 106.54   | 11.26 |
| tile_n   | YOLOv8n Tile Detection   | 35.37    | 27.20    | 46.36    | 28.27 |

**Testing**: 150 NUnit tests with 0 failures, including unit tests, integration tests, and mocking for all core services.

## Project Structure

```
vision-inspection-portfolio/
├── 01_training/                    # Python training pipeline
│   ├── app.py                     # Streamlit main application
│   ├── src/
│   │   ├── data_manager.py        # Dataset handling and preprocessing
│   │   ├── model_trainer.py       # YOLOv8 training logic
│   │   ├── evaluator.py           # Model evaluation and metrics
│   │   ├── exporter.py            # ONNX export functionality
│   │   └── gradcam.py             # Custom GradCAM implementation
│   ├── data/                      # MVTec dataset storage
│   └── outputs/                   # Training results and models
├── 02_inspection/                  # C# inspection system
│   ├── InspectionSystem.Core/      # Business logic layer
│   │   ├── Services/
│   │   │   ├── OnnxInferenceEngine.cs
│   │   │   ├── ImageProcessor.cs
│   │   │   └── SessionLogger.cs
│   │   └── Models/
│   ├── InspectionSystem.UI/        # Avalonia UI application
│   │   ├── ViewModels/            # MVVM view models
│   │   └── Views/                 # UI views and controls
│   ├── InspectionSystem.Tests/     # Comprehensive test suite
│   ├── ModelBenchmark/            # Console benchmark tool
│   └── models/                    # ONNX model files
└── README.md
```

## How to Run

**Program 1 - Model Trainer:**
```bash
cd 01_training
pip install -r requirements.txt
streamlit run app.py
```

**Program 2 - Inspection System:**
```bash
cd 02_inspection
dotnet run --project InspectionSystem.UI/InspectionSystem.UI.csproj
```

**Benchmark Tool:**
```bash
cd 02_inspection
dotnet run --project ModelBenchmark/ModelBenchmark.csproj
```

## Key Design Decisions

- **ONNX Runtime Integration**: Chose ONNX for model deployment to eliminate Python dependency in production C# environment, ensuring consistent performance across platforms
- **Custom GradCAM Implementation**: Built explainable AI features using PyTorch hooks instead of external libraries for better control over feature visualization and reduced dependencies
- **SOLID Architecture**: Implemented dependency injection and interface segregation in C# system for maintainable, testable codebase suitable for industrial environments
- **Avalonia UI Framework**: Selected cross-platform UI framework over WPF to support deployment on Linux manufacturing systems commonly used in semiconductor facilities
- **Comprehensive Testing Strategy**: Established 150+ unit and integration tests with mocking to ensure reliability in production environments where system failures are costly
- **CPU-Optimized Inference**: Focused on CPU performance optimization for deployment in environments where GPU resources may not be available or cost-prohibitive

## Target Applications

This portfolio demonstrates capabilities relevant to:
- Semiconductor manufacturing inspection (SFA, Hanwha)
- Machine vision systems (Cognex, Keyence)
- Industrial automation and robotics (Doosan, Rainbow)
- Quality control and defect detection systems

## Requirements

### Program 1 (Model Trainer)
- Python 3.11
- See 01_training/requirements.txt for full list

### Program 2 (Inspection System)
- .NET 10 SDK
- macOS Intel (OpenCvSharp4.runtime.osx.10.15-x64)
- ONNX model file at 02_inspection/models/bottle/best.onnx

## Quick Start

### Program 1
```bash
cd 01_training
source .venv/bin/activate
streamlit run app/main.py
```

### Program 2
```bash
cd 02_inspection
dotnet run --project InspectionSystem.UI/InspectionSystem.UI.csproj
```

### Run Tests
```bash
cd 02_inspection
dotnet test InspectionSystem.Tests/InspectionSystem.Tests.csproj
```

## Demo

### Program 1 Features
- Data tab: MVTec dataset visualization with bounding box overlays
- Train tab: YOLOv8 hyperparameter configuration and training progress
- Eval tab: Confusion matrix, PR curves, FP/FN case analysis
- Export tab: ONNX model conversion and inference benchmarking
- GradCAM tab: Custom hook-based feature visualization across 9 YOLOv8 layers

### Program 2 Features
- Inspection view: Real-time ONNX inference with bounding box rendering
- Benchmark view: Multi-model performance comparison and session statistics
- GradCAM view: Heatmap visualization with original/overlay display modes
- Settings view: Model configuration and inference parameter tuning
