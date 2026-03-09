## Program 2 Architecture (C# SOLID Design)

### Project Structure
InspectionSystem/
├── InspectionSystem.Core/          # Domain logic (no UI dependency)
│   ├── Interfaces/
│   │   ├── IInferenceEngine.cs     # O,D principle
│   │   ├── IGradCamService.cs
│   │   ├── IImageProcessor.cs
│   │   ├── ISessionLogger.cs
│   │   └── ISettingsService.cs
│   ├── Models/
│   │   ├── DetectionResult.cs
│   │   ├── InspectionSession.cs
│   │   ├── DefectClass.cs
│   │   └── AppSettings.cs
│   └── Services/
│       ├── OnnxInferenceEngine.cs  # implements IInferenceEngine
│       ├── GradCamService.cs       # implements IGradCamService
│       ├── ImageProcessor.cs       # implements IImageProcessor
│       ├── SessionLogger.cs        # implements ISessionLogger
│       └── SettingsService.cs      # implements ISettingsService
├── InspectionSystem.UI/            # Avalonia UI layer
│   ├── ViewModels/
│   │   ├── MainViewModel.cs
│   │   ├── InspectionViewModel.cs
│   │   ├── GradCamViewModel.cs
│   │   └── SettingsViewModel.cs
│   ├── Views/
│   │   ├── MainWindow.axaml
│   │   ├── InspectionView.axaml
│   │   ├── GradCamView.axaml
│   │   └── SettingsView.axaml
│   ├── Controls/                   # Custom Avalonia controls
│   │   ├── ImageCanvas.axaml       # BBox overlay canvas
│   │   └── HeatmapControl.axaml    # GradCAM heatmap
│   └── App.axaml                   # DI container setup
└── InspectionSystem.Tests/         # NUnit test project
    ├── Core/
    │   ├── OnnxInferenceEngineTests.cs
    │   ├── ImageProcessorTests.cs
    │   ├── GradCamServiceTests.cs
    │   └── SessionLoggerTests.cs
    └── ViewModels/
        ├── InspectionViewModelTests.cs
        └── SettingsViewModelTests.cs

### UI Design System (FluentAvalonia)
- FluentAvalonia NuGet package for Fluent Design controls
- Color scheme: dark theme with accent colors
- NavigationView for main navigation
- Cards for result panels
- Consistent spacing and typography
- Responsive layout with Grid and StackPanel

### WEEK 5 (Day 29~35) - Program 2 Foundation (SOLID + NUnit)
- Day 29 - TODO: Project restructure - Core / UI / Tests separation + DI setup
- Day 30 - TODO: Define all interfaces (IInferenceEngine, IImageProcessor, IGradCamService)
- Day 31 - TODO: OnnxInferenceEngine implementation + NUnit tests
- Day 32 - TODO: ImageProcessor implementation + NUnit tests
- Day 33 - TODO: GradCamService (C# calls Python FastAPI) + NUnit tests
- Day 34 - TODO: SessionLogger + SettingsService + NUnit tests
- Day 35 - TODO: DI container setup + integration test + GitHub push

### WEEK 6 (Day 36~42) - Program 2 MVVM + FluentAvalonia UI
- Day 36 - TODO: FluentAvalonia setup + NavigationView main layout
- Day 37 - TODO: InspectionViewModel + InspectionView (image panel + BBox overlay)
- Day 38 - TODO: Result panel (OK/NG / type / confidence / speed) with Fluent cards
- Day 39 - TODO: Params panel (Confidence / IoU sliders) + ViewModel binding
- Day 40 - TODO: GradCamViewModel + GradCamView (Split view original/heatmap)
- Day 41 - TODO: Session statistics bar + CSV export + button bindings
- Day 42 - TODO: ROI drag selection custom control + NUnit tests

### WEEK 7 (Day 43~49) - Advanced Features + Optimization
- Day 43 - TODO: Multithread inference (Task + CancellationToken) + NUnit tests
- Day 44 - TODO: Inference speed benchmark (target 30fps) + performance tests
- Day 45 - TODO: NG image auto-save + file management + NUnit tests
- Day 46 - TODO: Settings screen (model swap / save path) + ViewModel tests
- Day 47 - TODO: Edge case testing (empty image / multiple defects / model errors)
- Day 48 - TODO: Program 2 full integration test + NUnit coverage report
- Day 49 - TODO: Buffer + GitHub push

## Program 2 C# Coding Standards

### SOLID Examples in This Project
- Single Responsibility:
  OnnxInferenceEngine only handles ONNX inference
  SessionLogger only handles logging
  ImageProcessor only handles image preprocessing

- Open/Closed:
  New model formats (TensorRT, OpenVINO) can be added by implementing IInferenceEngine
  without modifying existing code

- Interface Segregation:
  IInferenceEngine: RunInference() only
  ISessionLogger: Log(), GetHistory(), ExportCsv() only
  Not one large IService interface

- Dependency Inversion:
  ViewModels receive IInferenceEngine via constructor injection
  Never instantiate OnnxInferenceEngine directly in ViewModel

### NUnit Test Naming Convention
Format: MethodName_StateUnderTest_ExpectedBehavior
Example:
  RunInference_ValidImage_ReturnsDetectionResult
  RunInference_NullImage_ThrowsArgumentNullException
  RunInference_EmptyImage_ReturnsEmptyDetections

### FluentAvalonia UI Guidelines
- Use FluentAvalonia NavigationView for main navigation
- Use Card control for result panels
- Use InfoBar for OK/NG status display
- Dark theme as default
- Accent color: #0078D4 (Windows Fluent blue)
- All colors defined in App.axaml resources (no hardcoded colors)