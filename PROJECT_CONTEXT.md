# Vision Inspection Portfolio - Project Context

## Project Overview
AI-based surface defect inspection system for job portfolio.
Target companies: Semiconductor equipment (SFA, Hanwha), Machine vision (Cognex, Keyence), Robotics (Doosan, Rainbow)

## Tech Stack
- Python 3.11 / PyTorch / YOLOv8 / ONNX Runtime
- C# .NET 10 / Avalonia UI / FluentAvalonia / CommunityToolkit.Mvvm
- GradCAM (custom PyTorch hook implementation - no library)
- MVTec AD Dataset (15 categories, 5,354 images)
- NUnit for all C# unit tests

## System Architecture

### Program 1: Model Trainer (Python + Streamlit UI) - COMPLETE
- YOLOv8 fine-tuning on MVTec AD bottle category
- Custom GradCAM implementation with PyTorch hooks (register_forward_hook + register_full_backward_hook)
- Confusion Matrix / PR Curve / F1 Curve / FP/FN case analysis
- ONNX export and speed benchmark
- 5 tabs: Data / Train / Eval / Export / GradCAM

### Program 2: Inspection System (C# + Avalonia UI) - IN PROGRESS (Day 30)
- Real-time ONNX Runtime inference (no Python needed)
- BBox overlay rendering with OpenCvSharp4
- GradCAM heatmap visualization (calls Python FastAPI)
- Session statistics + CSV export
- NG image auto-save
- Multithreaded inference (Task + SemaphoreSlim)
- SOLID principles strictly applied
- MVVM pattern with full separation of concerns
- NUnit unit tests for every feature

## Repository
URL: https://github.com/TaeyangYeon/vision-inspection-portfolio
Local: ~/vision-inspection-portfolio/
Git: TaeyangYeon / acrobatyeon@gmail.com

## Folder Structure
```
vision-inspection-portfolio/
├── 01_training/
│   ├── data/
│   │   ├── raw/                    # MVTec AD dataset (15 categories)
│   │   └── processed/bottle/       # YOLO format (images/train, images/val, labels/train, labels/val)
│   ├── models/best.onnx            # Trained model opset 21 (12.3MB)
│   ├── outputs/bottle_full/        # results.csv, eval_results.json, pr_data.json
│   ├── scripts/
│   │   ├── explore_data.py
│   │   ├── convert_to_yolo.py
│   │   └── visualize_labels.py
│   ├── app/
│   │   ├── main.py
│   │   ├── pages/
│   │   │   ├── data_tab.py
│   │   │   ├── train_tab.py
│   │   │   ├── eval_tab.py
│   │   │   ├── export_tab.py
│   │   │   └── gradcam_tab.py
│   │   ├── components/
│   │   │   └── augmentation.py
│   │   └── utils/
│   │       ├── model_utils.py
│   │       ├── fpfn_utils.py
│   │       ├── data_loader.py
│   │       ├── health_check.py
│   │       └── integration_test.py
│   ├── gradcam/
│   │   ├── gradcam_core.py
│   │   ├── gradcam_yolo.py
│   │   ├── gradcam_visualize.py
│   │   ├── gradcam_multiclass.py
│   │   ├── gradcam_test.py
│   │   └── README.md
│   └── .venv/
├── 02_inspection/
│   ├── models/best.onnx            # opset 21 (copied from 01_training)
│   ├── InspectionSystem.sln
│   ├── InspectionSystem.Core/
│   │   ├── Interfaces/
│   │   │   ├── IInferenceEngine.cs
│   │   │   ├── IImageProcessor.cs
│   │   │   ├── IGradCamService.cs
│   │   │   ├── ISessionLogger.cs
│   │   │   └── ISettingsService.cs
│   │   ├── Models/
│   │   │   ├── DetectionResult.cs
│   │   │   ├── InspectionRecord.cs
│   │   │   ├── SessionStats.cs
│   │   │   ├── AppSettings.cs
│   │   │   ├── ProcessedImage.cs
│   │   │   ├── GradCamResult.cs
│   │   │   ├── InferenceOptions.cs
│   │   │   └── DrawOptions.cs
│   │   └── Services/
│   │       ├── OnnxInferenceEngine.cs
│   │       ├── ImageProcessor.cs
│   │       ├── SessionLogger.cs
│   │       ├── SettingsService.cs
│   │       ├── GradCamService.cs
│   │       └── NgImageSaver.cs
│   ├── InspectionSystem.UI/
│   │   ├── App.axaml / App.axaml.cs
│   │   ├── DependencyInjection/
│   │   │   └── ServiceCollectionExtensions.cs
│   │   ├── ViewModels/
│   │   │   ├── MainViewModel.cs
│   │   │   ├── InspectionViewModel.cs
│   │   │   ├── GradCamViewModel.cs
│   │   │   ├── SettingsViewModel.cs
│   │   │   └── BenchmarkViewModel.cs
│   │   └── Views/
│   │       ├── MainWindow.axaml / MainWindow.axaml.cs
│   │       ├── InspectionView.axaml / InspectionView.axaml.cs
│   │       ├── GradCamView.axaml / GradCamView.axaml.cs
│   │       ├── SettingsView.axaml / SettingsView.axaml.cs
│   │       └── BenchmarkView.axaml / BenchmarkView.axaml.cs
│   └── InspectionSystem.Tests/
│       ├── Core/
│       │   ├── OnnxInferenceEngineTests.cs
│       │   ├── ImageProcessorTests.cs
│       │   ├── SessionLoggerTests.cs
│       │   ├── SettingsServiceTests.cs
│       │   ├── GradCamServiceTests.cs
│       │   └── NgImageSaverTests.cs
│       ├── ViewModels/
│       │   ├── InspectionViewModelTests.cs
│       │   └── SettingsViewModelTests.cs
│       └── Integration/
│           └── DiContainerTests.cs
└── PROJECT_CONTEXT.md
```

## Environment
- Mac (Intel), Python 3.11 (pyenv), .NET 10 SDK
- Streamlit run: `cd ~/vision-inspection-portfolio/01_training && source .venv/bin/activate && streamlit run app/main.py`
- Local URL: http://localhost:8501
- Training: Google Colab T4 GPU
- DNS: 8.8.8.8 (Google DNS) - required for NuGet connectivity on this machine

## NuGet Packages

### InspectionSystem.Core
- Microsoft.Extensions.DependencyInjection 10.0.3
- Microsoft.Extensions.Logging.Abstractions 10.0.3
- Microsoft.ML.OnnxRuntime 1.20.1
- OpenCvSharp4 4.13.0.20260308
- OpenCvSharp4.runtime.osx.10.15-x64 4.6.0.20230105 (Intel Mac specific)

### InspectionSystem.UI
- Avalonia / Avalonia.Themes.Fluent
- FluentAvalonia
- CommunityToolkit.Mvvm
- Microsoft.Extensions.DependencyInjection 10.0.3

### InspectionSystem.Tests
- NUnit 4.5.1
- NUnit3TestAdapter 6.1.0
- Microsoft.NET.Test.Sdk 18.3.0
- Moq 4.20.72
- coverlet.collector 6.0.4

## Training Results (Bottle Category)
- Model: YOLOv8n, 100 epochs, Colab T4 GPU
- mAP50: 0.8692 / mAP50-95: 0.677 / Precision: 0.807 / Recall: 0.800
- Class results: broken_large 0.912 / broken_small 0.898 / contamination 0.798
- ONNX export: best.onnx 12.3MB opset 21
- Inference speed: ~36ms / ~27 FPS (CPU, Intel Mac)
- Note: Model was re-trained in full 100 epochs after initial export had opset 22 issue.
  Re-exported with opset=21 for OnnxRuntime 1.20.1 compatibility.

## Test Status (as of Day 30)
- Total NUnit tests: 93 (all PASS, 0 fail, 0 skip)
- OnnxInferenceEngineTests: 8 tests
- ImageProcessorTests: 8 tests
- SessionLoggerTests: 13 tests (includes 2 CSV export tests)
- SettingsServiceTests: 8 tests
- GradCamServiceTests: 6 tests
- NgImageSaverTests: 10 tests
- InspectionViewModelTests: 13 tests
- SettingsViewModelTests: 9 tests
- DiContainerTests: 9 tests (includes BenchmarkViewModel)

## Program 2 Architecture

### Navigation Pages (5 pages)
- Inspection: Main defect detection view (Load Model / Open Image / Run Inspection)
- GradCAM: Heatmap visualization (3-column: original / heatmap / overlay)
- Benchmark: Inference speed benchmark + session records + CSV export
- Settings: Model path / threshold / save path / GradCAM config

### UI Design (Dark Theme - Catppuccin Mocha palette)
- Background: #1E1E2E / Surface: #181825 / Overlay: #313244
- Text: #CDD6F4 / Subtext: #BAC2DE / Muted: #6C7086
- OK color: #A6E3A1 (green) / NG color: #F38BA8 (red) / Accent: #0078D4

### SOLID Design Principles Applied
- S: Each class has one reason to change (NgImageSaver only saves, SessionLogger only logs)
- O: New model formats via IInferenceEngine without modifying existing code
- I: Small focused interfaces per feature (5 interfaces total)
- D: ViewModels receive interfaces via DI constructor injection
- MVVM: Model / ViewModel / View fully separated, no code-behind logic

### NUnit Naming Convention
- MethodName_StateUnderTest_ExpectedBehavior

## Known Issues and Solutions

### Issue 1: Python file encoding error (UnicodeDecodeError)
- Cause: Claude Code generates files with non-UTF-8 encoding when emojis are included
- Solution: Never use emojis in Python source files
- Prevention: Always add to Claude Code prompts:
  "Use plain text only, no emojis anywhere in the file content. Save with UTF-8 encoding."

### Issue 2: dataset.yaml uses local Mac absolute path
- Cause: convert_to_yolo.py saves absolute local path
- Solution: After uploading to Colab, overwrite yaml path with Colab path

### Issue 3: YOLO training mAP was 0.02 on first run
- Cause: All defect images went to val/ with none in train/
- Solution: Fixed split logic - defect images now 80% train / 20% val

### Issue 4: Streamlit use_column_width deprecated
- Solution: Replace with use_container_width=True

### Issue 5: GradCAM CAM Statistics NameError (sc2 not defined)
- Cause: st.columns() variable unpacking goes out of scope in nested with blocks
- Solution: Use list indexing - cam_stats = st.columns(3) / with cam_stats[0]:

### Issue 6: ONNX opset 22 not supported by OnnxRuntime 1.20.1
- Symptom: Model loads but produces zero detections silently
- Cause: Default Ultralytics export uses opset 22, OnnxRuntime only guarantees up to opset 21
- Solution: Re-export with opset=21
  model.export(format='onnx', opset=21, simplify=True, imgsz=640)
- Note: This issue caused 93x lower confidence (0.006 vs 0.63) before fix

### Issue 7: OpenCvSharp4 runtime package for Intel Mac
- Wrong: OpenCvSharp4.runtime.osx_arm64 (ARM only)
- Correct: OpenCvSharp4.runtime.osx.10.15-x64 (Intel Mac)

### Issue 8: Mat constructor not accessible in OpenCvSharp4
- Solution: Use Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData)

### Issue 9: GetArray<byte> fails for 3-channel Mat
- Solution: Use GetArray<Vec3b> then manually convert to byte[]

### Issue 10: NuGet DNS resolution failure
- Cause: Default DNS 208.67.220.123 could not resolve api.nuget.org
- Solution: Change macOS DNS to 8.8.8.8 in System Settings > Wi-Fi > Details > DNS

### Issue 11: C# preprocessing confidence 93x lower than Python
- Cause: ConvertBitmapToRgbBytes() was saving bitmap as encoded PNG bytes instead of raw pixels
- Symptom: Mat.FromPixelData received encoded image data causing memory corruption (SIGBUS exit code 138)
- Fix: Extract raw RGB pixels using OpenCV ImDecode + BGR2RGB + pixel array
- Also fixed: letterbox padding fill value must be 114 (not 0), top-left placement

### Issue 12: AppSettings model path uses relative path
- Symptom: Load Model fails silently with "Model not found"
- Fix: AppSettings.GetDefaultModelPath() scans multiple candidate paths at runtime
  and returns the first existing absolute path

## Key Technical Decisions
- GradCAM: custom PyTorch hook (no library) - stronger interview talking point
- ONNX Runtime in C#: Python-free inference - unique selling point vs typical Python apps
- Avalonia UI: cross-platform WPF alternative, works on Intel Mac
- MVTec AD: industry standard anomaly detection benchmark dataset
- .NET 10: latest SDK with full Avalonia support
- Intel Mac: training on Google Colab T4 GPU, inference locally via ONNX CPU
- SemaphoreSlim(1,1): prevents concurrent inference calls, guards against UI double-click
- CancellationToken: passed through all async inference paths for clean cancellation

## Daily Progress

### Day 1 - DONE
Python environment setup, pyenv + Python 3.11, venv, requirements.txt, GitHub init

### Day 2 - DONE
.NET 10 SDK, Avalonia templates, InspectionSystem project created, Hello World run

### Day 3 - DONE
MVTec AD dataset downloaded, explore_data.py + convert_to_yolo.py + visualize_labels.py

### Day 4 - DONE
Colab notebook, 10 epoch test run, mAP50: 0.4755

### Day 5 - DONE
100 epoch full training, mAP50: 0.8692, augmentation experiments

### Day 6 - DONE
ONNX export best.onnx 12.3MB, saved to both 01_training/models and 02_inspection/models

### Day 7 - DONE
Code cleanup, Streamlit environment verified

### Day 8 - DONE
Streamlit app structure, 4-tab navigation: Data / Train / Eval / Export

### Day 9 - DONE
Data tab: image viewer with BBox overlay, class distribution chart

### Day 10 - DONE
Augmentation preview (8 types) with interactive sliders

### Day 11 - DONE
Train tab: parameter form, training command builder, results viewer (Loss + mAP charts)

### Day 12 - DONE
Eval tab: Confusion Matrix, PR Curve, F1 Curve, per-class metrics (all Plotly interactive)

### Day 13 - DONE
Eval tab: sample image inference viewer with real ONNX model, confidence/IoU sliders

### Day 14 - DONE
Eval tab: FP/FN case analysis viewer - TP(green)/FP(red)/FN(blue), filter, batch analysis

### Day 15 - DONE
Export tab: model info, ONNX speed benchmark (36.2ms/27.6FPS), PT vs ONNX comparison

### Day 16 - DONE
Program 1 UI polish: health check system, sidebar improvements, tab descriptions

### Day 17 - DONE
Program 1 integration test all passed (20/20), GradCAM structure designed

### Day 18 - DONE
GradCAM core with PyTorch hooks, YOLO wrapper, test 2/2 passed

### Day 19 - DONE
GradCAM visualizer (9 layers), Streamlit GradCAM tab (5 tabs total)

### Day 20 - DONE
GradCAM multi-class comparison, All Classes mode, Program 1 complete

### Day 21 - DONE
Program 2 restructured to Core/UI/Tests, 5 interfaces + 8 models defined, Core builds clean
Fixed DNS issue (8.8.8.8), all NuGet packages installed

### Day 22 - DONE
OnnxInferenceEngine + ImageProcessor implemented, NUnit 20/20 PASS
Fixed: opset 21 re-export, OpenCvSharp4.runtime.osx.10.15-x64, Mat.FromPixelData, Vec3b GetArray

### Day 23 - DONE
SessionLogger + SettingsService implemented, all NUnit tests pass

### Day 24 - DONE
DI container, GradCamService (FastAPI HTTP client), integration tests for DI, all pass

### Day 25 - DONE
Avalonia MainWindow with dark theme, left navigation, session stats panel, status bar

### Day 26 - DONE
InspectionViewModel + InspectionView: image panel, OK/NG badge, detection list, sliders

### Day 27 - DONE
GradCamView + SettingsView implemented. All 4 pages navigating correctly.

### Day 28 - DONE
ViewModel NUnit tests (InspectionViewModel + SettingsViewModel), NgImageSaver with tests, 91 pass

### Day 29 - DONE
Multithreaded inference (Task + SemaphoreSlim), NG auto-save integrated into InspectionViewModel
Fixed critical bugs:
- Preprocessing BGR->RGB + letterbox padding 114 (93x confidence improvement)
- SIGBUS crash from encoded bytes passed to Mat.FromPixelData (raw pixel extraction fix)
- Model re-trained 100 epochs + re-exported opset 21 (original export was opset 22)
All 91 NUnit tests pass 0 fail 0 skip

### Day 30 - DONE
BenchmarkView (speed stats + run history bar chart), CSV export UI
SessionLogger ExportToCsvAsync added, ISessionLogger interface updated
BenchmarkViewModel + DI registration + MainWindow navigation added
All NUnit tests pass 0 fail 0 skip

## Remaining Plan

### Day 31 - DONE: Edge case testing (null/empty input, zero dimensions, cancelled token, model not loaded), OnnxInferenceEngine error handling hardened, integration tests expanded, all NUnit tests pass 0 fail 0 skip
### Day 32 - DONE: IntegrationScenarioTests (8 scenarios), NUnit coverage report generated, all tests pass 0 fail 0 skip
### Day 33 - TODO: Buffer + GitHub push + code review
### Day 34 - TODO: Multi-category support (tile/carpet training on Colab)
### Day 35 - TODO: Model ensemble experiment (YOLOv8n vs YOLOv8s comparison)
### Day 36 - TODO: Inference speed benchmark by model size (FPS table)
### Day 37 - TODO: Small defect detection improvement experiments
### Day 38 - TODO: Full performance metrics summary (mAP / FPS / accuracy by category)
### Day 39 - TODO: Buffer + GitHub push
### Day 40 - TODO: README writing (architecture diagram + performance metrics table)
### Day 41 - TODO: Demo video recording (Program 1 flow + Program 2 NG detection)
### Day 42 - TODO: GitHub Wiki - GradCAM custom implementation explanation
### Day 43 - TODO: Resume one-liner + interview prep key talking points
### Day 44 - TODO: Final GitHub push + portfolio submission ready

---
Last updated: Day 32 complete
Next: Day 33 - Buffer + GitHub cleanup + code review
---
