import sys
import numpy as np
import cv2
import torch
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent))

def run_multiclass_comparison(
    model_path: str,
    image: np.ndarray,
    class_names: dict,
    output_dir: Path,
    alpha: float = 0.5,
):
    from gradcam_yolo import run_gradcam

    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    results = {}
    for cls_id, cls_name in class_names.items():
        print(f"  Generating CAM for class {cls_id}: {cls_name}")
        cam, overlay, cam_resized = run_gradcam(
            model_path, image,
            target_class=int(cls_id),
            alpha=alpha,
        )
        if cam is not None:
            results[cls_id] = {
                "name": cls_name,
                "cam": cam,
                "overlay": overlay,
                "cam_resized": cam_resized,
                "mean": float(cam.mean()),
                "max": float(cam.max()),
            }
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"class_{cls_id}_{cls_name}_overlay.jpg"), overlay_bgr)
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(str(output_dir / f"class_{cls_id}_{cls_name}_heatmap.jpg"), heatmap)
            print(f"    mean={cam.mean():.4f} max={cam.max():.4f}")
        else:
            print(f"    [SKIP] CAM generation failed")

    return results

def build_comparison_grid(image: np.ndarray, results: dict, output_dir: Path):
    h, w = image.shape[:2]
    cell_w, cell_h = 320, 320
    n_classes = len(results)
    n_cols = n_classes + 1
    canvas_w = n_cols * cell_w
    canvas_h = cell_h * 2

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    orig_resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (cell_w, cell_h)
    )
    canvas[:cell_h, :cell_w] = orig_resized
    cv2.putText(canvas, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for idx, (cls_id, data) in enumerate(results.items()):
        col = idx + 1
        x = col * cell_w

        overlay_bgr = cv2.cvtColor(data["overlay"], cv2.COLOR_RGB2BGR)
        overlay_resized = cv2.resize(overlay_bgr, (cell_w, cell_h))
        canvas[:cell_h, x:x+cell_w] = overlay_resized

        label = f"{data['name']} (mean:{data['mean']:.3f})"
        cv2.putText(canvas, label, (x + 8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        heatmap = cv2.applyColorMap(
            (data["cam_resized"] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_resized = cv2.resize(heatmap, (cell_w, cell_h))
        canvas[cell_h:, x:x+cell_w] = heatmap_resized

        row2_label = f"max:{data['max']:.3f}"
        cv2.putText(canvas, row2_label, (x + 8, cell_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(canvas, "Heatmap", (10, cell_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    save_path = output_dir / "multiclass_comparison.jpg"
    cv2.imwrite(str(save_path), canvas)
    print(f"Comparison grid saved: {save_path}")
    return canvas

def print_summary(results: dict):
    print("\n=== Multi-class GradCAM Summary ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
    for cls_id, data in sorted_results:
        bar_len = int(data["mean"] * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"  Class {cls_id} ({data['name']:15s}): [{bar}] mean={data['mean']:.4f}")
    print()
    top = sorted_results[0]
    print(f"Highest activation: Class {top[0]} ({top[1]['name']}) mean={top[1]['mean']:.4f}")

if __name__ == "__main__":
    import yaml

    pt_path = BASE_PATH / "models/best.pt"
    if not pt_path.exists():
        print("ERROR: best.pt not found at models/best.pt")
        exit(1)

    yaml_path = BASE_PATH / "data/processed/bottle/dataset.yaml"
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    class_names = yaml_data.get("names", {})

    val_dir = BASE_PATH / "data/processed/bottle/images/val"
    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    defect_images = [p for p in images if not p.stem.isdigit()]

    if not defect_images:
        print("No defect images found.")
        exit(1)

    test_images = defect_images[:3]
    print(f"Testing {len(test_images)} images with {len(class_names)} classes")

    for img_path in test_images:
        print(f"\nImage: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        output_dir = BASE_PATH / f"outputs/gradcam_multiclass/{img_path.stem}"
        results = run_multiclass_comparison(
            str(pt_path), img_rgb, class_names, output_dir
        )
        build_comparison_grid(img_rgb, results, output_dir)
        print_summary(results)

    print("\nDone. Check outputs/gradcam_multiclass/")