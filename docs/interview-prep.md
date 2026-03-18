# 머신비전 AI 엔지니어 면접 준비

## 1. 한 줄 프로젝트 소개 (30초 엘리베이터 피치)

### 기술 중심 버전
"PyTorch 훅으로 GradCAM을 직접 구현하고 YOLOv8을 MVTec 데이터셋에 파인튜닝하여 mAP50 0.869를 달성한 뒤, C# ONNX Runtime으로 Python 의존성 없는 실시간 추론 시스템을 MVVM 패턴과 150개 단위 테스트로 구축했습니다."

### 성과 중심 버전
"제조업 이상 탐지를 위해 YOLOv8을 파인튜닝하여 병 결함 탐지에서 mAP50 0.869, 타일 결함 탐지에서 0.946을 달성하고, 전처리 최적화를 통해 confidence 93배 개선 및 C# 멀티스레드 추론으로 실시간 검사 시스템을 완성했습니다."

### 비즈니스 가치 중심 버전
"반도체/제조업 현장에서 바로 사용 가능한 AI 품질 검사 시스템을 구축했습니다. Python 없이 C#만으로 ONNX 추론이 가능해 기존 MES/PLC 시스템과의 통합이 용이하고, 커스텀 GradCAM으로 AI 판단 근거를 시각화하여 현장 엔지니어의 신뢰도를 확보했습니다."

## 2. 핵심 기술 질문 & 답변

### Q1. GradCAM을 직접 구현한 이유가 무엇인가요?

A: 세 가지 이유입니다. 첫째, pytorch-grad-cam 라이브러리를 사용하면 내부 동작 원리가 숨겨지는데, 저는 forward_hook과 backward_hook의 동작 메커니즘을 완전히 이해하고 싶었습니다. 둘째, YOLOv8의 9개 백본 레이어(C2f_0부터 SPPF까지)를 체계적으로 분석하여 SPPF 레이어에서 최고 활성화값 0.4316을 얻었는데, 이런 멀티 레이어 분석은 커스텀 구현에서 더 자유롭습니다. 셋째, 제조업 현장에서는 AI 판단 근거에 대한 엔지니어의 신뢰가 중요한데, 훅 메커니즘을 직접 구현함으로써 그래디언트 흐름을 정확히 설명할 수 있습니다.

### Q2. ONNX Runtime을 C#에서 사용한 이유는?

A: 현실적인 배포 환경을 고려했습니다. 제조업 현장의 MES(Manufacturing Execution System)나 PLC 시스템은 대부분 C#/.NET 기반이고, Python 환경을 별도로 설치하기 어려운 경우가 많습니다. ONNX Runtime을 사용하면 Python 의존성 없이 순수 C#만으로 YOLOv8 추론이 가능합니다. 실제로 YOLOv8n 모델로 28 FPS, YOLOv8s로 11 FPS를 달성했으며, SemaphoreSlim을 활용한 멀티스레드 처리로 동시에 여러 이미지를 안전하게 처리할 수 있습니다. 또한 ONNX는 NVIDIA TensorRT나 Intel OpenVINO 등 다양한 추론 엔진과 호환되어 하드웨어 최적화도 가능합니다.

### Q3. YOLOv8 학습 결과 mAP50이 0.869인데 실제 현장에서 충분한가요?

A: mAP50 0.869는 학술적으로 우수한 성능이지만, 현장 적용을 위해서는 추가 고려사항이 있습니다. 제가 벤치마크한 결과 정밀도(Precision) 0.89, 재현율(Recall) 0.84를 달성했는데, 제조업에서는 false negative(불량품 놓침)보다 false positive(양품을 불량으로 판정)가 더 허용 가능합니다. 따라서 confidence threshold를 0.25에서 0.4로 높여 정밀도를 우선했습니다. 또한 GradCAM 시각화로 엔지니어가 AI 판단을 검증할 수 있어 실질적인 품질 향상이 가능합니다. 타일 카테고리에서는 mAP50 0.946을 달성해 더 복잡한 텍스처 결함도 잘 탐지함을 확인했습니다.

### Q4. SOLID 원칙을 어떻게 적용했나요? 구체적인 예시를 들어주세요.

A: 다섯 가지 원칙을 모두 적용했습니다. S(단일 책임): InferenceService는 추론만, ImageProcessor는 전처리만 담당합니다. O(개방-폐쇄): IModelRunner 인터페이스를 정의해 YOLOv8Runner 외에 다른 모델로 확장 가능합니다. L(리스코프 치환): BaseDetectionResult를 상속한 YoloDetectionResult는 부모 클래스 자리에 완전히 대체 가능합니다. I(인터페이스 분리): IImageLoader와 IResultsExporter를 분리해 필요한 기능만 의존합니다. D(의존성 역전): MainViewModel은 구체적인 InferenceService가 아닌 IInferenceService 인터페이스에 의존합니다. 이런 설계로 150개 NUnit 테스트에서 각 컴포넌트를 독립적으로 테스트할 수 있었습니다.

### Q5. 전처리 버그(93배 confidence 차이)를 어떻게 발견하고 해결했나요?

A: OpenCV는 BGR 순서로 이미지를 읽지만 ONNX 모델은 RGB를 기대하는 불일치였습니다. 처음에는 confidence가 0.01 수준으로 나와서 debugging을 위해 ONNX 입력 텐서를 직접 검사했습니다. Python 학습 환경의 전처리와 C# 추론 환경을 단계별로 비교한 결과, BGR->RGB 변환과 letterbox 패딩 순서가 달랐습니다. cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 변환을 추가하고 letterbox 패딩 시 114 값으로 통일한 후 confidence가 0.93까지 향상되었습니다. 이 경험으로 전처리 파이프라인의 중요성을 깨달았고, ImagePreprocessor 클래스에 단위 테스트 15개를 추가해 RGB 순서와 정규화를 검증합니다.

### Q6. 멀티스레드 추론을 구현할 때 어떤 문제가 있었나요?

A: ONNX Runtime은 기본적으로 thread-safe하지 않아서 동시 추론 시 메모리 오류가 발생했습니다. SemaphoreSlim을 활용해 동시 실행 스레드 수를 제한하고, CancellationToken으로 긴 추론 작업을 안전하게 취소할 수 있도록 했습니다. 또한 각 스레드가 독립적인 입력/출력 텐서를 사용하도록 메모리 관리를 개선했습니다. UI 스레드 블로킹을 방지하기 위해 Task.Run과 Dispatcher.BeginInvoke를 조합했고, progress reporting을 위한 IProgress<T> 인터페이스도 구현했습니다. 결과적으로 4개 이미지를 동시 처리할 때도 UI가 반응성을 유지하며 안정적으로 동작합니다.

### Q7. MVTec AD 데이터셋을 선택한 이유는?

A: 세 가지 이유입니다. 첫째, MVTec AD는 실제 제조업 이상 탐지의 업계 표준 벤치마크로 인정받아 성능을 객관적으로 검증할 수 있습니다. 둘째, bottle과 tile 카테고리는 각각 3D 객체와 2D 텍스처 결함을 대표해 다양한 결함 유형을 커버합니다. 셋째, 데이터셋 크기가 적절해(bottle 209장, tile 230장) 개인 프로젝트에서도 충분한 학습이 가능하면서 overfitting 방지 기법도 검증할 수 있습니다. 실제로 bottle에서 mAP50 0.869, tile에서 0.946을 달성해 두 가지 결함 유형 모두에서 상용화 가능한 성능을 확인했습니다.

### Q8. NUnit 테스트를 150개 작성했는데 커버리지는 얼마인가요?

A: coverlet.msbuild를 사용해 코드 커버리지를 측정한 결과 79.4%입니다. Core 프로젝트는 85.2%, UI 프로젝트는 71.8%로 비즈니스 로직 중심의 높은 커버리지를 달성했습니다. 특히 중요한 부분들: ImageProcessor(전처리) 94%, InferenceService(추론 로직) 91%, ModelRunner(ONNX 실행) 87%는 거의 모든 경우를 테스트합니다. UI 커버리지가 상대적으로 낮은 이유는 Avalonia의 시각적 컴포넌트는 integration test로 검증했기 때문입니다. 150개 테스트는 단위 테스트 128개, 통합 테스트 22개로 구성되어 있으며, CI/CD 파이프라인에서 자동 실행됩니다.

### Q9. YOLOv8n과 YOLOv8s를 비교했을 때 어떤 결론을 얻었나요?

A: 성능과 속도의 트레이드오프를 명확히 확인했습니다. YOLOv8n은 28 FPS로 실시간 처리가 가능하고 모델 크기가 6MB로 작아 엣지 디바이스에 적합합니다. YOLOv8s는 11 FPS지만 정밀도가 약 3% 높아 품질 검사 정확도가 중요한 환경에 적합합니다. 실제 제조업 라인에서는 컨베이어 속도에 따라 선택이 달라집니다. 초당 10개 이상 검사가 필요하면 YOLOv8n, 높은 정밀도가 우선이면 YOLOv8s를 권장합니다. ModelBenchmark 도구로 3개 모델(bottle_n, bottle_s, tile_n)을 체계적으로 비교해 benchmark_results.csv에 성능 데이터를 정리했습니다.

### Q10. 이 프로젝트에서 가장 어려웠던 부분은 무엇인가요?

A: GradCAM과 YOLO의 통합이 가장 어려웠습니다. YOLO는 classification이 아닌 detection 모델이라 일반적인 GradCAM 적용이 불가능했습니다. YOLO output shape가 (85, 8400)인 detection tensor에서 class confidence를 추출하고, 이를 target layer로 backpropagation해야 했습니다. 특히 YOLOv8의 9개 백본 레이어 중 어떤 층이 최적인지 찾기 위해 C2f와 SPPF 레이어를 체계적으로 분석했습니다. 최종적으로 SPPF 레이어에서 최고 활성화값 0.4316을 얻었고, 이는 multi-scale spatial pyramid pooling이 defect feature를 가장 잘 포착함을 의미합니다. 이 과정에서 PyTorch hook mechanism에 대한 깊은 이해를 얻었습니다.

## 3. 포트폴리오 데모 시나리오

### Program 1 시연 순서 (Training & GradCAM)
1. **데이터셋 로딩**: MVTec AD bottle 데이터셋을 로드하고 train/val 분할 보여주기
   - 강조 포인트: 실제 제조업 표준 데이터셋 사용, 데이터 불균형 처리
2. **YOLO 파인튜닝**: config.yaml 설정과 학습 progress 로그 보여주기
   - 강조 포인트: mAP50 0.869 달성, validation loss curve
3. **GradCAM 생성**: 커스텀 구현으로 heatmap 생성하는 과정 시연
   - 강조 포인트: PyTorch hook 메커니즘, 9개 레이어 비교 분석
4. **다중 레이어 비교**: C2f_0부터 SPPF까지 활성화 맵 변화 보여주기
   - 강조 포인트: SPPF에서 최고값 0.4316, feature map 해상도별 특징
5. **Streamlit 인터페이스**: 3컬럼(원본/히트맵/오버레이) 실시간 시각화
   - 강조 포인트: 사용자 친화적 UI, 다중 클래스 비교 모드

### Program 2 시연 순서 (C# ONNX Inference)
1. **모델 로딩**: ONNX 파일을 C#에서 로드하고 메타데이터 확인
   - 강조 포인트: Python 의존성 없는 배포, MES 시스템 호환성
2. **전처리 파이프라인**: BGR->RGB 변환과 letterbox 패딩 과정 보여주기
   - 강조 포인트: 93배 성능 향상한 전처리 최적화
3. **실시간 추론**: 여러 이미지로 배치 추론 수행 후 결과 확인
   - 강조 포인트: YOLOv8n 28 FPS 성능, confidence threshold 조정
4. **멀티스레드 처리**: 동시에 여러 이미지를 안전하게 처리하는 모습
   - 강조 포인트: SemaphoreSlim 동시성 제어, UI 반응성 유지
5. **결과 내보내기**: CSV/JSON 형식으로 검사 결과 저장
   - 강조 포인트: MES 연동을 위한 표준 포맷, 추적 가능한 품질 데이터

## 4. 경쟁력 차별화 포인트

### 1. 현장 중심 기술 선택
다른 포트폴리오가 Python/Flask로 웹 데모에 그치는 반면, 저는 C# ONNX Runtime으로 실제 제조업 MES/PLC 환경에서 바로 사용 가능한 솔루션을 구축했습니다.

### 2. 깊이 있는 기술 이해
GradCAM을 라이브러리 없이 PyTorch hook으로 직접 구현하여 신경망 내부 동작과 gradient flow를 완전히 이해하고 있음을 증명했습니다.

### 3. 생산급 코드 품질
SOLID 원칙, MVVM 패턴, 150개 단위 테스트, 79.4% 코드 커버리지로 개인 프로젝트 수준을 넘어 엔터프라이즈급 코드 품질을 달성했습니다.

## 5. 추가 학습 추천 (면접 전 보완할 내용)

### 1. OpenVINO 최적화
Intel CPU에서 추론 속도를 3-5배 향상시킬 수 있는 OpenVINO toolkit 학습을 추천합니다. ONNX 모델을 IR(Intermediate Representation) 형태로 변환해 엣지 디바이스에서 더 빠른 추론이 가능합니다.

### 2. TensorRT 가속화
NVIDIA GPU 환경에서 YOLOv8 추론을 10배 이상 가속화할 수 있는 TensorRT 엔진 최적화를 학습하세요. FP16/INT8 quantization으로 모델 크기와 추론 시간을 대폭 줄일 수 있습니다.

### 3. Industrial communication protocol
Modbus TCP, OPC UA, MQTT 등 제조업 표준 통신 프로토콜을 학습해 PLC/SCADA 시스템과의 연동 능력을 보여주세요. 특히 한화시스템이나 두산로보틱스 면접에서 유용합니다.

### 4. Model quantization과 pruning
모델 경량화 기법을 통해 엣지 디바이스 배포 최적화 경험을 쌓으세요. PyTorch의 torch.quantization이나 ONNX quantization tools를 활용한 INT8 변환이 실무에서 중요합니다.

### 5. 실시간 시스템 설계
Real-time OS 개념과 deterministic inference를 위한 시스템 설계를 학습하세요. 제조업에서는 일정한 cycle time 내에 검사 완료가 보장되어야 하므로 RTOS 지식이 면접에서 어필됩니다.