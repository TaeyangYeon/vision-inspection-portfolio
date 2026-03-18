# Program 2: AI Inspection System - Complete Description

## What is this program?

This is a real-time surface defect inspection system built with C# and Avalonia UI.

It loads a trained AI model and inspects product images instantly.
No Python or AI framework installation is required to run this program.

The purpose of this program is to simulate a production-grade inspection system
that an engineer could deploy in a real manufacturing environment.

---

# Background Knowledge

## What is ONNX Runtime?

ONNX Runtime is a library that runs AI models without requiring Python or PyTorch.

When an AI model is trained in Python, it is exported to ONNX format.
ONNX Runtime can then load and run that model in any programming language,
including C#, C++, and Java.

Benefits of ONNX Runtime

- No Python installation needed on the deployment machine
- Faster inference through graph optimization
- Works on Windows, macOS, and Linux
- Supported by Microsoft

This program loads the ONNX model trained in Program 1
and runs inference entirely in C#.

---

## What is C#?

C# is a programming language developed by Microsoft.

It is widely used in

- industrial automation systems
- Windows desktop applications
- machine control software
- embedded systems

Many machine vision systems in manufacturing use C# because it integrates
well with hardware SDKs and industrial communication protocols.

This project uses C# with .NET 10, the latest version of the .NET platform.

---

## What is Avalonia UI?

Avalonia is a cross-platform UI framework for C# applications.

It works on Windows, macOS, and Linux.

It is similar to WPF (Windows Presentation Foundation) but runs on all platforms.

This project uses Avalonia because the development machine is an Intel Mac,
and Avalonia allows building a native desktop application without Windows.

---

## What is MVVM?

MVVM stands for Model-View-ViewModel.

It is a design pattern that separates the user interface from the business logic.

Three layers

Model
The data and core logic (DetectionResult, InspectionRecord, AppSettings)

ViewModel
Connects the model to the UI. Handles button clicks and state changes.

View
The XAML UI layout. Only displays data, contains no logic.

Why MVVM matters

- UI can be changed without touching business logic
- Business logic can be tested without running the UI
- Easier to maintain as the project grows

This project applies MVVM strictly.
All ViewModels are tested with NUnit.
No logic exists in the View code-behind files.

---

## What is SOLID?

SOLID is a set of five design principles for writing maintainable software.

S - Single Responsibility
Each class has only one reason to change.
Example: NgImageSaver only saves images. SessionLogger only logs records.

O - Open/Closed
Classes are open for extension but closed for modification.
Example: Adding a new model format only requires a new IInferenceEngine implementation.
Existing code does not change.

L - Liskov Substitution
Any implementation can replace its interface without breaking the system.

I - Interface Segregation
Interfaces are small and focused.
Example: IInferenceEngine only handles inference.
ISessionLogger only handles logging.
They are never combined into one large interface.

D - Dependency Inversion
High-level classes depend on interfaces, not concrete implementations.
Example: InspectionViewModel receives IInferenceEngine via constructor injection.
It never creates OnnxInferenceEngine directly.

---

## What is Dependency Injection?

Dependency Injection is a technique where objects receive their dependencies
from outside rather than creating them internally.

Without Dependency Injection

    var engine = new OnnxInferenceEngine();
    var vm = new InspectionViewModel(engine);

With Dependency Injection

    services.AddTransient<IInferenceEngine, OnnxInferenceEngine>();
    services.AddTransient<InspectionViewModel>();
    var vm = provider.GetRequiredService<InspectionViewModel>();

Benefits

- Easy to swap implementations (e.g. replace ONNX with TensorRT)
- Easy to test by injecting mock objects
- Clear ownership of object lifetimes

This project uses Microsoft.Extensions.DependencyInjection,
the same DI framework used in ASP.NET Core production servers.

---

## What is NUnit?

NUnit is a unit testing framework for C#.

A unit test verifies that a small piece of code behaves correctly.

Example

    [Test]
    public void SessionLogger_LogRecord_IncreasesTotalCount()
    {
        _logger.Log(new InspectionRecord { IsNG = true });
        Assert.That(_logger.GetStats().TotalCount, Is.EqualTo(1));
    }

This project contains 150 NUnit tests covering

- Core services (OnnxInferenceEngine, ImageProcessor, SessionLogger)
- ViewModels (InspectionViewModel, SettingsViewModel)
- Integration scenarios (DI container, CSV export, NG auto-save)

All 150 tests pass with 0 failures.

---

## What is OpenCvSharp?

OpenCvSharp is a C# wrapper for OpenCV.

OpenCV is the most widely used computer vision library in the world.

This project uses OpenCvSharp for

- Image preprocessing (resize, padding, color conversion)
- Drawing bounding boxes on detected defects
- Converting between image formats

---

## What is Letterbox Padding?

YOLOv8 requires all input images to be exactly 640x640 pixels.

If an image is resized directly to 640x640 without preserving the aspect ratio,
the image becomes distorted.

Letterbox padding solves this by

1. Scaling the image so the longest side becomes 640
2. Padding the shorter side with gray pixels (value 114)
3. Placing the image in the top-left corner

This preserves the original proportions and improves detection accuracy.

This project fixed a critical bug where letterbox padding was implemented
incorrectly in C#.
The incorrect implementation produced confidence values 93 times lower
than the Python reference implementation.
Correcting the padding increased max confidence from 0.006 to 0.632.

---

## What is GradCAM in Program 2?

Program 2 includes a GradCAM visualization tab.

When running GradCAM in Program 2

1. The image is sent to a Python FastAPI server running Program 1
2. The server generates the heatmap using the custom GradCAM implementation
3. The heatmap is returned to Program 2 and displayed in a 3-column view

This architecture allows Program 2 to display GradCAM results
without reimplementing the PyTorch hook logic in C#.

---

## What is Multithreaded Inference?

Running inference on the UI thread would freeze the application
while the model is processing.

This project runs inference on a background thread using Task.Run.

Key mechanisms used

Task.Run
Runs the inference on a thread pool thread.
The UI remains responsive during processing.

SemaphoreSlim(1, 1)
Prevents two inference calls from running simultaneously.
If the user clicks Run Inspection twice quickly,
the second call is rejected immediately.

CancellationToken
Allows the user to cancel a running inference.
The token is checked before inference starts
and passed through the entire call chain.

---

## What is CSV Export?

After running multiple inspections, the session data can be exported to CSV.

The CSV file contains

- Timestamp
- Image path
- Result (OK or NG)
- Defect count
- Inference time (ms)
- Confidence score

This allows engineers to analyze inspection results in Excel
or import them into a quality management system.

---

# Project Overview

## System Pipeline

Program 2 follows a standard industrial inspection workflow.

1. Load Model
   Load the ONNX model into memory via ONNX Runtime

2. Open Image
   Load a product image from disk

3. Preprocess
   Apply letterbox padding and normalize pixel values

4. Run Inference
   Pass the tensor through ONNX Runtime

5. Parse Output
   Extract bounding boxes, class IDs, and confidence scores

6. Draw Results
   Overlay colored bounding boxes on the original image

7. Log Record
   Save the result to session history

8. Auto-save NG
   If the result is NG, automatically save the image to disk

---

## Goal

Simulate a production-grade AI inspection system that demonstrates
C# engineering skills alongside AI knowledge.

The target roles for this portfolio are

- Machine vision engineer
- AI software engineer
- Inspection system developer

at companies such as SFA, Hanwha, Cognex, Keyence, Doosan, and Rainbow.

---

# Inspection Results

## OK Result

No defects detected in the image.

The result badge displays OK in green.

## NG Result

One or more defects detected.

The result badge displays NG in red.
Detected defects are listed with class name, confidence score, and bounding box.
If auto-save is enabled, the result image is saved automatically.

---

# Inference Performance

| Metric | Value |
|--------|-------|
| Inference time (avg) | ~35 ms |
| Estimated FPS | ~28 |
| Model size (YOLOv8n) | 12.3 MB |
| Model size (YOLOv8s) | 44.7 MB |
| Runtime | CPU (Intel Mac) |

Running on CPU is important for industrial deployment
because many inspection machines do not have a dedicated GPU.

---

# Application Pages

## Inspection Page

The main inspection workflow page.

Features

- Load Model button
- Open Image button
- Run Inspection button
- Cancel button
- Original image display
- Result image with bounding boxes
- OK / NG result badge
- Detection list (class name, confidence, bounding box)
- Inference time display
- Confidence and IoU threshold sliders

---

## GradCAM Page

Visualizes which regions the AI model focused on.

Features

- Open Image button
- Generate GradCAM button
- Target class selector
- Opacity slider
- 3-column view: Original / Heatmap / Overlay
- CAM Mean and CAM Max statistics
- API connection status indicator

---

## Benchmark Page

Measures and displays inference speed.

Features

- Run Benchmark button
- Warmup run count selector
- Benchmark run count selector
- Speed statistics: Min / Avg / Max / FPS
- Per-run progress bar chart
- Session record list
- CSV export with configurable path

---

## Settings Page

Configures all system parameters.

Features

- Model path input
- Confidence threshold
- IoU threshold
- Image size
- NG image save path
- Auto-save NG toggle
- GradCAM API URL
- Enable GradCAM toggle
- Save Settings button
- Reset Defaults button

---

# Software Architecture

## Project Structure

The C# solution is divided into three projects.

InspectionSystem.Core
Contains all business logic.
No dependency on UI framework.
Fully testable.

InspectionSystem.UI
Contains all Avalonia XAML views and ViewModels.
Depends on Core via interfaces only.

InspectionSystem.Tests
Contains all 150 NUnit tests.
Tests Core services and ViewModels independently.

## Interface Design

Five interfaces define the system contracts.

IInferenceEngine
Loads the ONNX model and runs inference.

IImageProcessor
Preprocesses images and draws detection results.

ISessionLogger
Records inspection results and exports to CSV.

ISettingsService
Loads and saves application settings.

IGradCamService
Sends images to the GradCAM API and returns heatmaps.

---

# Key Technical Highlights

## Preprocessing Bug Fix

The C# preprocessing pipeline initially produced confidence values
93 times lower than the Python reference.

Root cause: ConvertBitmapToRgbBytes was saving the bitmap as encoded PNG bytes
instead of extracting raw pixel data.
This caused memory corruption when OpenCV tried to create a Mat from encoded bytes.

Fix applied

1. Use OpenCV ImDecode to load raw bytes correctly
2. Apply BGR to RGB color conversion
3. Apply letterbox scaling with fill value 114
4. Place scaled image in top-left corner of 640x640 canvas
5. Normalize pixel values to 0.0 to 1.0 range
6. Convert to NCHW tensor format

After the fix, max confidence improved from 0.006 to 0.632,
matching the Python reference exactly.

---

## Concurrent Inference Guard

If the user clicks Run Inspection multiple times quickly,
only the first call proceeds.

The second call is rejected immediately with a status message.

This is implemented using SemaphoreSlim(1, 1).WaitAsync(0).
The timeout of 0 means the check is non-blocking.

---

## NG Auto-save

When inference returns an NG result and auto-save is enabled,
the result image is saved to disk automatically.

File naming format

    NG_20240315_143022_456_broken_large.jpg

The filename includes timestamp and defect class name
so engineers can quickly identify the defect type from the filename alone.

---

# Glossary

| Term | Meaning |
|------|---------|
| ONNX | Universal AI model format |
| ONNX Runtime | Library that runs ONNX models without Python |
| MVVM | UI design pattern separating View and logic |
| SOLID | Five software design principles |
| DI | Dependency Injection |
| NUnit | C# unit testing framework |
| SemaphoreSlim | Thread synchronization primitive |
| CancellationToken | Mechanism for cancelling async operations |
| Letterbox | Aspect-ratio-preserving resize with padding |
| mAP50 | Object detection accuracy metric at IoU 0.5 |
| NG | Not Good - defect detected |
| OK | No defect detected |
| FPS | Frames Per Second - inference speed |
