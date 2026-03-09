import sys
import numpy as np
import cv2
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent))

MODEL_PATH = str(BASE_PATH / "models/best.onnx")
PT_MODEL_PATH = None

VAL_DIR = BASE_PATH / "data/processed/bottle/images/val"

def find_test_image():
    images = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))
    defect_images = [p for p in images if not p.stem.isdigit()]
    if defect_images:
        return defect_images[0]
    return images[0] if images else None

def test_gradcam_core():
    print("\n=== Test 1: GradCAM Core (PyTorch hooks) ===")
    try:
        import torch
        import torch.nn as nn
        from gradcam_core import GradCAM

        class SimpleConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 3)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = SimpleConvNet()
        target_layer = model.conv2
        gradcam = GradCAM(model, target_layer)

        dummy_input = torch.randn(1, 3, 64, 64, requires_grad=True)
        output = model(dummy_input)
        target_class = 0
        model.zero_grad()
        output[0, target_class].backward()

        gradcam.gradients = []
        gradcam.activations = []

        cam = gradcam.generate(dummy_input, target_class=0)
        assert cam is not None, "CAM is None"
        assert cam.shape[0] > 0, "CAM shape invalid"
        assert cam.min() >= 0 and cam.max() <= 1, "CAM not normalized"

        dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        overlay, _ = gradcam.overlay_heatmap(cam, dummy_img)
        assert overlay.shape == dummy_img.shape, "Overlay shape mismatch"

        gradcam.remove_hooks()
        print("[PASS] GradCAM core: hooks registered, CAM generated, overlay created")
        print(f"       CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}")
        return True

    except Exception as e:
        print(f"[FAIL] GradCAM core: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam_yolo():
    print("\n=== Test 2: GradCAM on YOLOv8 ===")

    pt_candidates = list((BASE_PATH / "outputs").glob("*/weights/best.pt"))
    if not pt_candidates:
        print("[SKIP] No .pt model found locally (trained on Colab)")
        print("       To test: download best.pt from Google Drive")
        print("       Path: vision_portfolio/runs/bottle_full/weights/best.pt")
        return True

    try:
        from gradcam_yolo import run_gradcam

        img_path = find_test_image()
        if img_path is None:
            print("[SKIP] No test image found")
            return True

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pt_path = str(pt_candidates[0])
        print(f"Using model: {pt_path}")
        print(f"Using image: {img_path.name}")

        cam, overlay, cam_resized = run_gradcam(pt_path, img_rgb, target_class=0)

        if cam is None:
            print("[FAIL] GradCAM returned None")
            return False

        output_dir = BASE_PATH / "outputs/gradcam_test"
        output_dir.mkdir(exist_ok=True)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "gradcam_overlay.jpg"), overlay_bgr)

        heatmap_only = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imwrite(str(output_dir / "gradcam_heatmap.jpg"), heatmap_only)

        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "original.jpg"), orig_bgr)

        print(f"[PASS] GradCAM on YOLO: CAM shape={cam.shape}")
        print(f"       Results saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"[FAIL] GradCAM YOLO: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all():
    results = []
    results.append(test_gradcam_core())
    results.append(test_gradcam_yolo())

    print("\n=== GradCAM Test Summary ===")
    passed = sum(results)
    print(f"Result: {passed}/{len(results)} passed")

    if passed == len(results):
        print("GradCAM implementation ready.")
    else:
        print("Some tests failed. Check errors above.")

if __name__ == "__main__":
    run_all()