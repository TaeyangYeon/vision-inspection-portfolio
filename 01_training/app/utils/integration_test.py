import sys
import traceback
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    tests = []
    modules = [
        "streamlit",
        "cv2",
        "numpy",
        "torch",
        "ultralytics",
        "onnxruntime",
        "plotly",
        "pandas",
        "yaml",
        "PIL",
    ]
    for mod in modules:
        try:
            __import__(mod)
            tests.append((mod, True, "OK"))
        except ImportError as e:
            tests.append((mod, False, str(e)))
    return tests

def test_data_pipeline():
    tests = []

    try:
        import cv2
        import numpy as np
        from utils.data_loader import get_image_list
        images = get_image_list("bottle", "val")
        assert len(images) > 0, "No val images found"
        img = cv2.imread(str(images[0]))
        assert img is not None, "Failed to load image"
        tests.append(("Data pipeline", True, f"OK ({len(images)} val images)"))
    except Exception as e:
        tests.append(("Data pipeline", False, str(e)))

    return tests

def test_onnx_inference():
    tests = []

    try:
        import onnxruntime as ort
        import numpy as np
        import cv2
        from utils.model_utils import preprocess_image, postprocess_output

        model_path = BASE_PATH / "models/best.onnx"
        assert model_path.exists(), "ONNX model not found"

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor, scale, orig_shape = preprocess_image(dummy)
        output = session.run(None, {input_name: tensor})

        assert output is not None, "Inference returned None"
        assert len(output) > 0, "Empty output"
        tests.append(("ONNX inference", True, f"OK (output shape: {output[0].shape})"))
    except Exception as e:
        tests.append(("ONNX inference", False, str(e)))

    return tests

def test_augmentation():
    tests = []

    try:
        import numpy as np
        import cv2
        from components.augmentation import apply_augmentations

        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        config = {
            "horizontal_flip": True,
            "brightness": True,
            "brightness_factor": 1.5,
            "gaussian_noise": True,
        }
        results = apply_augmentations(dummy, config)
        assert len(results) >= 3, "Not enough augmentation results"
        tests.append(("Augmentation", True, f"OK ({len(results)} transforms)"))
    except Exception as e:
        tests.append(("Augmentation", False, str(e)))

    return tests

def test_eval_data():
    tests = []

    try:
        import json
        eval_path = BASE_PATH / "outputs/bottle_full/eval_results.json"
        assert eval_path.exists(), "eval_results.json not found"
        with open(eval_path) as f:
            data = json.load(f)
        assert "mAP50" in data, "mAP50 not in eval data"
        assert "confusion_matrix" in data, "confusion_matrix not in eval data"
        tests.append(("Eval data", True, f"OK (mAP50: {data['mAP50']:.4f})"))
    except Exception as e:
        tests.append(("Eval data", False, str(e)))

    try:
        import json
        pr_path = BASE_PATH / "outputs/bottle_full/pr_data.json"
        assert pr_path.exists(), "pr_data.json not found"
        with open(pr_path) as f:
            data = json.load(f)
        tests.append(("PR data", True, "OK"))
    except Exception as e:
        tests.append(("PR data", False, str(e)))

    return tests

def run_all_tests():
    all_tests = []
    all_tests += test_imports()
    all_tests += test_data_pipeline()
    all_tests += test_onnx_inference()
    all_tests += test_augmentation()
    all_tests += test_eval_data()

    print("\n=== Program 1 Integration Test ===")
    passed = 0
    failed = 0
    for name, ok, status in all_tests:
        icon = "PASS" if ok else "FAIL"
        print(f"[{icon}] {name}: {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nResult: {passed}/{passed+failed} passed")
    if failed == 0:
        print("All tests passed. Program 1 is ready.")
    else:
        print(f"{failed} test(s) failed. Please fix before proceeding.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)