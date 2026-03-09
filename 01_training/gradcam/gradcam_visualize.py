import sys
import numpy as np
import cv2
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent))

def visualize_activation_maps(model_path: str, image: np.ndarray, output_dir: Path):
    import torch
    from ultralytics import YOLO
    from gradcam_core import GradCAM

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    pytorch_model = model.model
    pytorch_model.eval()

    backbone = pytorch_model.model
    target_layers = {}
    for i, layer in enumerate(backbone):
        layer_name = type(layer).__name__
        if layer_name in ["C2f", "C3", "SPPF"]:
            target_layers[f"layer_{i}_{layer_name}"] = layer

    print(f"Found {len(target_layers)} target layers: {list(target_layers.keys())}")

    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    tensor = torch.from_numpy(
        padded.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    tensor.requires_grad_(True)

    results = {}
    for layer_name, layer in target_layers.items():
        try:
            gradcam = GradCAM(pytorch_model, layer)
            cam = gradcam.generate(tensor, target_class=0)
            gradcam.remove_hooks()

            if cam is None:
                continue

            overlay, cam_resized = gradcam.overlay_heatmap(cam, image)

            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            cv2.imwrite(str(output_dir / f"{layer_name}_overlay.jpg"), overlay_bgr)
            cv2.imwrite(str(output_dir / f"{layer_name}_heatmap.jpg"), heatmap)

            results[layer_name] = {
                "cam": cam,
                "overlay": overlay,
                "cam_resized": cam_resized,
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
                "cam_mean": float(cam.mean()),
            }
            print(f"[OK] {layer_name}: cam shape={cam.shape} mean={cam.mean():.4f}")

        except Exception as e:
            print(f"[SKIP] {layer_name}: {e}")

    orig_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "original.jpg"), orig_bgr)

    return results

def compare_layers(results: dict, image: np.ndarray, output_dir: Path):
    if not results:
        print("No results to compare.")
        return

    h, w = image.shape[:2]
    cell_w, cell_h = 300, 300
    n_cols = min(len(results) + 1, 4)
    n_rows = (len(results) + 1 + n_cols - 1) // n_cols

    canvas = np.ones((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8) * 255

    orig_resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (cell_w, cell_h)
    )
    canvas[:cell_h, :cell_w] = orig_resized
    cv2.putText(canvas, "Original", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for idx, (layer_name, data) in enumerate(results.items()):
        col = (idx + 1) % n_cols
        row = (idx + 1) // n_cols
        x = col * cell_w
        y = row * cell_h

        overlay_bgr = cv2.cvtColor(data["overlay"], cv2.COLOR_RGB2BGR)
        overlay_resized = cv2.resize(overlay_bgr, (cell_w, cell_h))
        canvas[y:y+cell_h, x:x+cell_w] = overlay_resized

        short_name = layer_name.split("_")[-1]
        cv2.putText(canvas, short_name, (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(str(output_dir / "layer_comparison.jpg"), canvas)
    print(f"Layer comparison saved: {output_dir / 'layer_comparison.jpg'}")

if __name__ == "__main__":
    from pathlib import Path
    import cv2

    pt_path = BASE_PATH / "models/best.pt"
    if not pt_path.exists():
        print("ERROR: best.pt not found at models/best.pt")
        print("Download from Google Drive: vision_portfolio/runs/bottle_full/weights/best.pt")
        exit(1)

    val_dir = BASE_PATH / "data/processed/bottle/images/val"
    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    defect_images = [p for p in images if not p.stem.isdigit()]

    if not defect_images:
        print("No defect images found in val set.")
        exit(1)

    img_path = defect_images[0]
    print(f"Using image: {img_path.name}")

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    output_dir = BASE_PATH / "outputs/gradcam_visualization"
    results = visualize_activation_maps(str(pt_path), img_rgb, output_dir)
    compare_layers(results, img_rgb, output_dir)

    print(f"\nDone. Check results in: {output_dir}")