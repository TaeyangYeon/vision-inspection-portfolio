# Vision Inspection Portfolio - Project Context

## Project Overview
AI-based surface defect inspection system for job portfolio.
Target companies: Semiconductor equipment (SFA, Hanwha), Machine vision (Cognex, Keyence), Robotics (Doosan, Rainbow)

## Tech Stack
- Python 3.11 / PyTorch / YOLOv8 / ONNX Runtime
- C# .NET 10 / Avalonia UI
- GradCAM (custom PyTorch hook implementation - no library)
- MVTec AD Dataset (15 categories, 5,354 images)

## System Architecture
Program 1: Model Trainer (Python + Streamlit UI)
- YOLOv8 fine-tuning
- Custom GradCAM implementation with PyTorch hooks
- Confusion Matrix / PR Curve / FP/FN analysis
- ONNX export and verification

Program 2: Inspection System (C# + Avalonia UI)
- Real-time inference with ONNX Runtime (no Python needed)
- BBox overlay rendering
- GradCAM heatmap visualization (Split view)
- ROI drag selection
- Session statistics + CSV export
- SOLID principles strictly applied throughout
- MVVM pattern with full separation of concerns
- Avalonia UI with FluentAvalonia for modern Material-like design
- NUnit unit tests for every implemented feature

## Repository
URL: https://github.com/TaeyangYeon/vision-inspection-portfolio
Local: ~/vision-inspection-portfolio/

## Folder Structure
vision-inspection-portfolio/
├── 01_training/
│   ├── data/
│   │   ├── raw/          # MVTec AD dataset (15 categories)
│   │   └── processed/    # YOLO format converted data
│   ├── models/           # Trained model files (best.onnx)
│   ├── outputs/          # Training results (results.csv per run)
│   ├── scripts/
│   │   ├── explore_data.py       # Dataset structure analysis
│   │   ├── convert_to_yolo.py    # MVTec to YOLO format converter
│   │   └── visualize_labels.py   # Label verification visualizer
│   ├── app/
│   │   ├── main.py               # Streamlit entry point
│   │   ├── pages/
│   │   │   ├── data_tab.py       # Data management tab
│   │   │   ├── train_tab.py      # Training monitor tab
│   │   │   ├── eval_tab.py       # Evaluation results tab
│   │   │   └── export_tab.py     # ONNX export tab
│   │   ├── components/
│   │   │   ├── augmentation.py   # Augmentation preview
│   │   │   └── charts.py         # Reusable chart components
│   │   └── utils/
│   │       ├── data_loader.py    # Dataset loading utilities
│   │       └── model_utils.py    # Model related utilities
│   └── .venv/            # Python virtual environment
├── 02_inspection/
│   └── InspectionSystem/ # C# Avalonia project
└── PROJECT_CONTEXT.md

## nvironment
- Mac (Intel)
- Python 3.11 (pyenv)
- .NET 10 SDK
- Avalonia Templates installed
- Git configured: TaeyangYeon / acrobatyeon@gmail.com

## Streamlit App
- Run command: cd ~/vision-inspection-portfolio/01_training && source .venv/bin/activate && streamlit run app/main.py
- Local URL: http://localhost:8501

## Daily Progress

### Day 1 - DONE
- pyenv + Python 3.11 installed
- Project folder structure created
- venv created and activated
- requirements.txt created and all packages installed
- Environment verification script: scripts/check_env.py
- Git initialized and configured
- First commit pushed to GitHub

### Day 2 - DONE
- .NET 10 SDK confirmed (9 and 10 both installed)
- Avalonia templates installed
- InspectionSystem project created (avalonia.mvvm template)
- NuGet packages added: Avalonia, ONNX Runtime, OpenCvSharp4
- Hello World build and run successful
- Project folder structure organized (Models, Services, ViewModels, Views, Assets, Helpers)

### Day 3 - DONE
- MVTec AD dataset downloaded (Kaggle) and extracted
- 15 categories / 5,354 total images confirmed
- explore_data.py: dataset structure analysis script
- convert_to_yolo.py: MVTec mask to YOLO bbox conversion
  - cv2.findContours used for mask to bbox conversion
  - good images go to train/ with empty labels
  - defect images split 80% train / 20% val with bbox labels
  - dataset.yaml generated per category
- visualize_labels.py: bbox visualization verification
- Bottle category conversion verified successfully

### Day 4 - DONE
- Google Colab notebook created: 01_training/train_colab.ipynb
- Bottle dataset compressed and uploaded to Google Drive
- YOLOv8n 10 epoch test run completed
- mAP50: 0.4755 / Precision: 0.769 / Recall: 0.403

### Day 5 - DONE
- Augmentation experiments (Mosaic, Flip, HSV)
- Full training 100 epochs completed on Colab T4 GPU
- mAP50: 0.8692 / mAP50-95: 0.677 / Precision: 0.807 / Recall: 0.800
- Class results: broken_large 0.912 / broken_small 0.898 / contamination 0.798

### Day 6 - DONE
- ONNX export complete: best.onnx (12.3MB)
- Saved to: 01_training/models/best.onnx
- Saved to: 02_inspection/models/best.onnx

### Day 7 - DONE
- Code cleanup completed
- Streamlit environment verified
- check_env.py passed all checks

### Day 8 - DONE
- Streamlit app structure created (app/ folder with pages/, components/, utils/)
- Main navigation with 4 tabs: Data / Train / Eval / Export
- Sidebar project status panel
- IMPORTANT: Never use emojis in Python source files (causes UnicodeDecodeError)

### Day 9 - DONE
- Data tab: image viewer with BBox overlay
- Data tab: class distribution bar chart (Plotly)
- Category / split selector
- Show/hide labels toggle
- Random image button

### Day 10 - DONE
- Augmentation preview section added to Data tab
- 8 augmentation types: Horizontal Flip, Vertical Flip, Rotation, Brightness, HSV Shift, Gaussian Noise, Blur, Mosaic
- Interactive sliders for each augmentation parameter
- Fixed deprecated use_column_width to use_container_width

### Day 11 - DONE
- Train tab: parameter form (model size, epochs, batch, lr, augmentation settings)
- Train tab: training command builder (generates YOLO CLI command)
- Train tab: results viewer with Loss curves and mAP chart (Plotly)
- results.csv downloaded from Colab and placed in outputs/bottle_full/

## 60-Day Plan

### WEEK 1 (Day 1~7) - Environment + Data
- Day 1 - DONE: Python environment setup + GitHub init
- Day 2 - DONE: .NET + Avalonia setup
- Day 3 - DONE: MVTec dataset + YOLO conversion scripts
- Day 4 - DONE: YOLOv8 first training run - 10 epoch test mAP50 0.4755
- Day 5 - DONE: Augmentation experiments + Full training 100 epochs mAP50 0.8692
- Day 6 - DONE: ONNX export complete best.onnx 12.3MB saved to both projects
- Day 7 - DONE: Code cleanup Streamlit environment verified

### WEEK 2 (Day 8~14) - Program 1 Core UI
- Day 8 - DONE: Streamlit app structure and navigation layout
- Day 9 - DONE: Data tab image viewer with BBox overlay and class distribution chart
- Day 10 - DONE: Augmentation preview with 8 types and interactive controls
- Day 11 - DONE: Train tab parameter form training command builder results viewer
- Day 12 - DONE: Eval tab complete - Confusion Matrix, PR Curve, F1 Curve, per-class metrics (all Plotly interactive)
- Day 13 - DONE: Eval tab - sample image inference viewer with real ONNX model, confidence/IoU sliders, FP/FN detection
- Day 14 - DONE: Eval tab FP/FN case analysis viewer - TP/FP/FN classification with color coding, filter by case type, batch analysis on val set

### WEEK 3 (Day 15~21) - Program 1 Vision Depth
- Day 15 - DONE: Export tab complete - model info, ONNX speed benchmark, PT vs ONNX output comparison
- Day 16 - TODO: Export tab - ONNX conversion + PT vs ONNX result comparison
- Day 17 - DONE: Program 1 integration test all passed, GradCAM structure designed, data_loader utility created
- Day 18 - DONE: GradCAM core implemented with PyTorch hooks, YOLO wrapper created, test script passed
- Day 19 - DONE: GradCAM activation map visualizer + Streamlit GradCAM tab added (5 tabs total)
  - gradcam_visualize.py: 9 target layers analyzed (C2f + SPPF)
  - Layer progression: 160x160 (early) to 20x20 (deep) feature maps
  - SPPF layer showed highest activation (0.4316) for defect features
  - GradCAM tab added to Streamlit with 3-column view (original/heatmap/overlay)
  - Fixed: st.columns() NameError - use list indexing instead of variable unpacking
- Day 20 - DONE: GradCAM multi-class comparison, All Classes mode in Streamlit, Program 1 complete
- Day 21 - DONE: Program 2 restructured - Core/UI/Tests projects, all interfaces defined, all models defined, Core builds clean

### WEEK 4 (Day 22~28) - GradCAM Complete
- Day 22 - DONE: OnnxInferenceEngine + ImageProcessor implemented, NUnit 20/20 PASS (0 skip, 0 fail). Fixed: OpenCvSharp4.runtime.osx.10.15-x64, ONNX opset 21 re-export, Mat.FromPixelData + Vec3b GetArray
- Day 23 - DONE: SessionLogger + SettingsService implemented with full NUnit tests. Running total: all tests pass with 0 fail 0 skip
- Day 24 - DONE: DI container setup, GradCamService implemented, integration tests for DI resolution added. All NUnit tests pass 0 fail 0 skip.
- Day 25 - DONE: Avalonia MainWindow with dark theme, left navigation panel, session stats, status bar. DI fully wired to UI layer.
- Day 26 - DONE: InspectionViewModel + InspectionView with image panel, BBox overlay, OK/NG badge, detection list, confidence/IoU sliders
- Day 27 - TODO: GradCAM view + settings view implementation
- Day 28 - TODO: ONNX Runtime model load + single image inference test

### WEEK 5 (Day 29~35) - Program 2 Foundation (SOLID + NUnit)
- Day 29 - TODO: Project restructure - Core / UI / Tests separation + DI setup
- Day 30 - TODO: Define all interfaces (IInferenceEngine, IImageProcessor, IGradCamService)
- Day 31 - TODO: OnnxInferenceEngine implementation + NUnit tests
- Day 32 - TODO: ImageProcessor implementation + NUnit tests
- Day 33 - TODO: GradCamService (C# calls Python FastAPI) + NUnit tests
- Day 34 - TODO: SessionLogger + SettingsService + NUnit tests
- Day 35 - TODO: DI container setup + integration test + GitHub push

### WEEK 6 (Day 36~42) - Program 2 Avalonia UI
- Day 36 - TODO: Main window layout XAML
- Day 37 - TODO: Left panel - inspection image view (BBox overlay)
- Day 38 - TODO: Right panel - Result panel (OK/NG / type / confidence / speed)
- Day 39 - TODO: Right panel - Params panel (Confidence / IoU sliders)
- Day 40 - TODO: GradCAM tab - Split view (original / heatmap)
- Day 41 - TODO: Bottom - session statistics bar + button connections
- Day 42 - TODO: ROI drag selection feature

### WEEK 7 (Day 43~49) - Advanced Features + Optimization
- Day 43 - TODO: Inference speed measurement + bottleneck analysis (target 30fps)
- Day 44 - TODO: Multithread processing (prevent UI freezing)
- Day 45 - TODO: NG image auto-save + CSV export
- Day 46 - TODO: Settings screen (model swap / save path)
- Day 47 - TODO: Edge case testing (empty image / multiple defects)
- Day 48 - TODO: Program 2 full integration test
- Day 49 - TODO: Buffer + GitHub push

### WEEK 8 (Day 50~56) - Vision Expertise Depth
- Day 50 - TODO: Multi-category support (add tile/carpet training)
- Day 51 - TODO: Model ensemble experiment (YOLOv8n vs YOLOv8s comparison)
- Day 52 - TODO: Inference speed benchmark (FPS comparison table by model size)
- Day 53 - TODO: Small defect detection improvement (tile size experiment)
- Day 54 - TODO: Full performance metrics summary (mAP / FPS / accuracy)
- Day 55 - TODO: Buffer + GitHub push
- Day 56 - TODO: README writing (architecture diagram + performance metrics)

### WEEK 9 (Day 57~60) - Portfolio Complete
- Day 57 - TODO: Demo video recording (Program 1 to Program 2)
- Day 58 - TODO: GitHub Wiki - GradCAM custom implementation docs
- Day 59 - TODO: Resume one-liner + interview prep (key talking points)
- Day 60 - TODO: Final GitHub push + portfolio submission ready

## Known Issues and Solutions

### Issue 5: GradCAM CAM Statistics NameError (sc2 not defined)
- Symptom: NameError: name 'sc2' is not defined in gradcam_tab.py
- Cause: st.columns() unpacking variables go out of scope inside nested with blocks
- Solution: Use list indexing instead of variable unpacking
  Wrong:   sc1, sc2, sc3 = st.columns(3)
  Correct: cam_stats = st.columns(3)
           with cam_stats[0]: ...
           with cam_stats[1]: ...
           with cam_stats[2]: ...
- Prevention: Always use list indexing for st.columns() inside nested blocks

### Issue 1: Python file encoding error (UnicodeDecodeError)
- Symptom: UnicodeDecodeError utf-8 codec can't decode byte in Python files
- Cause: Claude Code generates files with non-UTF-8 encoding when emojis are included
- Solution: Never use emojis in Python source files
- Prevention: Always add this to Claude Code prompts when creating Python files:
  "Use plain text only, no emojis anywhere in the file content. Save with UTF-8 encoding."

### Issue 2: dataset.yaml path uses local Mac path
- Symptom: FileNotFoundError when running YOLO training on Colab
- Cause: convert_to_yolo.py saves absolute local path in dataset.yaml
- Solution: After uploading to Colab, overwrite dataset.yaml path with Colab path using Python
- Prevention: convert_to_yolo.py should use relative paths in dataset.yaml

### Issue 3: YOLO training mAP was 0.02 on first run
- Symptom: mAP50 = 0.0243 after 10 epochs
- Cause: convert_to_yolo.py was sending ALL defect images to val/ with no defects in train/
- Solution: Fixed split logic - defect images now split 80% train / 20% val

### Issue 4: Streamlit use_column_width deprecated warning
- Symptom: Yellow warning box above images in Streamlit
- Solution: Replace use_column_width=True with use_container_width=True

## Key Technical Decisions
- GradCAM: custom PyTorch hook implementation (no library) - stronger interview answer
- ONNX Runtime in C#: Python-free inference - unique selling point
- Avalonia UI: cross-platform WPF alternative for Mac development
- MVTec AD: industry standard anomaly detection benchmark dataset
- .NET 10: latest SDK with full Avalonia support
- Intel Mac: training on Google Colab T4 GPU, inference locally via ONNX CPU
- Program 2 C# Design Principles:
  - SOLID strictly applied:
    S - Single Responsibility: each class has one reason to change
    O - Open/Closed: open for extension closed for modification (interfaces + abstractions)
    I - Interface Segregation: small focused interfaces per feature
    D - Dependency Inversion: depend on abstractions not concrete classes (DI container)
  - MVVM pattern: Model / ViewModel / View fully separated
  - NUnit + test project for every feature (target 80%+ coverage)
  - FluentAvalonia for modern UI design (Material-like components)
  - Dependency Injection via Microsoft.Extensions.DependencyInjection

## How To Continue In New Chat
1. Upload this PROJECT_CONTEXT.md file
2. Say: "This is my vision inspection portfolio project context.
   I completed up to Day X. Please continue from Day X+1."
3. Claude will resume from exact current state.

---
Last updated: Day 26 complete
Next: Day 27 - GradCAM view + settings view implementation
---