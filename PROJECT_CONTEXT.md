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
- Confusion Matrix / PR Curve evaluation
- ONNX export and verification

Program 2: Inspection System (C# + Avalonia UI)
- Real-time inference with ONNX Runtime (no Python needed)
- BBox overlay rendering
- GradCAM heatmap visualization (Split view)
- ROI drag selection
- Session statistics + CSV export

## Repository
URL: https://github.com/TaeyangYeon/vision-inspection-portfolio
Local: ~/vision-inspection-portfolio/

## Folder Structure
vision-inspection-portfolio/
├── 01_training/
│   ├── data/
│   │   ├── raw/          # MVTec AD dataset (15 categories)
│   │   └── processed/    # YOLO format converted data
│   ├── models/           # Trained model files
│   ├── outputs/          # Training results
│   ├── scripts/
│   │   ├── explore_data.py       # Dataset structure analysis
│   │   ├── convert_to_yolo.py    # MVTec to YOLO format converter
│   │   └── visualize_labels.py   # Label verification visualizer
│   └── .venv/            # Python virtual environment
├── 02_inspection/
│   └── InspectionSystem/ # C# Avalonia project
└── PROJECT_CONTEXT.md

## Environment
- Mac (Intel)
- Python 3.11 (pyenv)
- .NET 10 SDK
- Avalonia Templates installed
- Git configured: TaeyangYeon / acrobatyeon@gmail.com

## Daily Progress

### Day 1 ✅
- pyenv + Python 3.11 installed
- Project folder structure created
- venv created and activated
- requirements.txt created and all packages installed
- Environment verification script: scripts/check_env.py
- Git initialized and configured
- First commit pushed to GitHub

### Day 2 ✅
- .NET 10 SDK confirmed (9 and 10 both installed)
- Avalonia templates installed
- InspectionSystem project created (avalonia.mvvm template)
- NuGet packages added: Avalonia, ONNX Runtime, OpenCvSharp4
- Hello World build and run successful
- Project folder structure organized (Models, Services, ViewModels, Views, Assets, Helpers)

### Day 3 ✅
- MVTec AD dataset downloaded (Kaggle) and extracted
- 15 categories / 5,354 total images confirmed
- explore_data.py: dataset structure analysis script
- convert_to_yolo.py: MVTec mask → YOLO bbox conversion
  - cv2.findContours used for mask to bbox conversion
  - good images → train/ with empty labels
  - defect images → val/ with bbox labels
  - dataset.yaml generated per category
- visualize_labels.py: bbox visualization verification
- Bottle category conversion verified successfully

## 60-Day Plan

### WEEK 1 (Day 1~7) - Environment + Data
- Day 1 ✅ Python environment setup + GitHub init
- Day 2 ✅ .NET + Avalonia setup
- Day 3 ✅ MVTec dataset + YOLO conversion scripts
- Day 4 ✅ YOLOv8 first training run - 10 epoch test, mAP50: 0.4755
- Day 5 ✅ Augmentation experiments + Full training 100 epochs, mAP50: 0.8692
- Day 6 ✅ ONNX export complete (best.onnx 12.3MB, saved to both projects)
- Day 7 ✅ Code cleanup, Streamlit environment verified

### WEEK 2 (Day 8~14) - Training Pipeline
- Day 8 ✅ Streamlit app structure created, navigation layout complete
  NOTE: Encoding issue fixed - Claude Code generates files with non-UTF-8 bytes when using emojis.
  Solution: Never use emojis in any Python source files. Use plain text only.
- Day 9 ✅ Data tab complete - image viewer with BBox overlay + class distribution chart
- Day 10 ✅ Augmentation preview added to Data tab - 8 augmentation types with interactive controls
- Day 11 ⬜ Program 1 Streamlit basic layout
- Day 12 ⬜ Data tab - image load + label overlay view
- Day 13 ⬜ Data tab - Augmentation preview
- Day 14 ⬜ Data tab - class distribution chart

### WEEK 3 (Day 15~21) - Program 1 UI
- Day 15 ⬜ Train tab - parameter input form
- Day 16 ⬜ Train tab - start/stop button connection
- Day 17 ⬜ Train tab - Loss curve realtime update
- Day 18 ⬜ Train tab - mAP realtime graph
- Day 19 ⬜ Eval tab - Confusion Matrix rendering
- Day 20 ⬜ Eval tab - PR Curve rendering
- Day 21 ⬜ Export tab - ONNX conversion + PT vs ONNX comparison

### WEEK 4 (Day 22~28) - Program 1 Complete + GradCAM Start
- Day 22 ⬜ Program 1 full flow test + bug fixes
- Day 23 ⬜ Buffer + GitHub push
- Day 24 ⬜ GradCAM theory + PyTorch hook concept code
- Day 25 ⬜ forward hook implementation (activation map)
- Day 26 ⬜ backward hook implementation (gradient extraction)
- Day 27 ⬜ CAM weight calculation + heatmap generation
- Day 28 ⬜ YOLO layer target + heatmap overlay

### WEEK 5 (Day 29~35) - GradCAM Complete + Program 2 Start
- Day 29 ⬜ Library result vs custom implementation comparison
- Day 30 ⬜ Buffer + GradCAM stabilization
- Day 31 ⬜ Avalonia project structure + NuGet packages
- Day 32 ⬜ ONNX Runtime model load + single image inference
- Day 33 ⬜ Inference result parsing (BBox / class / confidence)
- Day 34 ⬜ BBox overlay rendering on image
- Day 35 ⬜ Video file frame extraction + continuous inference

### WEEK 6 (Day 36~42) - Program 2 UI
- Day 36 ⬜ FastAPI server setup (GradCAM call)
- Day 37 ⬜ Main window layout XAML
- Day 38 ⬜ Left panel - inspection image view (BBox overlay)
- Day 39 ⬜ Right panel - Result panel (OK/NG / type / confidence / speed)
- Day 40 ⬜ Right panel - Params panel (Confidence / IoU sliders)
- Day 41 ⬜ GradCAM tab - Split view (original / heatmap)
- Day 42 ⬜ Buffer + GitHub push

### WEEK 7 (Day 43~49) - Advanced Features
- Day 43 ⬜ Bottom - session statistics bar + button connections
- Day 44 ⬜ ROI drag selection feature
- Day 45 ⬜ Inference speed measurement + bottleneck analysis (target 30fps)
- Day 46 ⬜ Multithread processing (prevent UI freezing)
- Day 47 ⬜ NG image auto-save + CSV export
- Day 48 ⬜ Settings screen (model swap / save path)
- Day 49 ⬜ Edge case testing (empty image / multiple defects)

### WEEK 8 (Day 50~56) - Optimization + Stabilization
- Day 50 ⬜ Full integration test
- Day 51 ⬜ Buffer + GitHub push
- Day 52 ⬜ README writing (architecture diagram + performance metrics)
- Day 53 ⬜ Demo video recording (Program 1 → Program 2)
- Day 54 ⬜ GitHub cleanup (commit history / Wiki / tech decision docs)
- Day 55 ⬜ Resume one-liner + final check
- Day 56 ⬜ Final buffer

### WEEK 9 (Day 57~60) - Portfolio Complete
- Day 57 ⬜ Final performance metrics: mAP / FPS / inference time
- Day 58 ⬜ Interview preparation: key talking points per feature
- Day 59 ⬜ Final GitHub push + repository cleanup
- Day 60 ⬜ Portfolio submission ready

## Known Issues & Solutions

### Issue 1: Python file encoding error (UnicodeDecodeError)
- Symptom: UnicodeDecodeError utf-8 codec can't decode byte in Python files
- Cause: Claude Code generates files with non-UTF-8 encoding when emojis are included in source code
- Solution: Never use emojis in Python source files (main.py, data_tab.py, etc.)
- Prevention: When asking Claude Code to create Python files, always add this instruction:
  "Use plain text only, no emojis anywhere in the file content"

### Issue 2: dataset.yaml path uses local Mac path
- Symptom: FileNotFoundError when running YOLO training on Colab
- Cause: convert_to_yolo.py saves absolute local path in dataset.yaml
- Solution: After uploading to Colab, overwrite dataset.yaml path with Colab path using Python
- Prevention: convert_to_yolo.py should use relative paths in dataset.yaml

### Issue 3: YOLO training mAP was 0.02 on first run
- Symptom: mAP50 = 0.0243 after 10 epochs
- Cause: convert_to_yolo.py was sending ALL defect images to val/ and good images to train/
  Result: train set had 0 defect samples for model to learn from
- Solution: Fixed split logic - defect images now split 80% train / 20% val

## Key Technical Decisions
- GradCAM: custom PyTorch hook (no library) → stronger interview answer
- ONNX Runtime in C#: Python-free inference → unique selling point
- Avalonia UI: cross-platform WPF alternative for Mac development
- MVTec AD: industry standard anomaly detection benchmark dataset
- .NET 10: latest SDK, full Avalonia support

## How To Continue In New Chat
1. Upload this PROJECT_CONTEXT.md file
2. Say: "This is my vision inspection portfolio project context.
   I completed up to Day X. Please continue from Day X+1."
3. Claude will resume from exact current state.

---
Last updated: Day 10 complete
Next: Day 11 - Train tab implementation