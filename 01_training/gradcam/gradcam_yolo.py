import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from gradcam_core import GradCAM

def get_target_layer(model: YOLO):
    backbone = model.model.model
    for i in range(len(backbone) - 1, -1, -1):
        layer = backbone[i]
        layer_name = type(layer).__name__
        if layer_name in ["C2f", "C3", "SPPF"]:
            print(f"Target layer found: index={i}, type={layer_name}")
            return layer
    print("Fallback: using second to last layer")
    return backbone[-2]

def preprocess_for_gradcam(image: np.ndarray, img_size: int = 640):
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = torch.from_numpy(tensor).unsqueeze(0)
    tensor.requires_grad_(True)
    return tensor, scale, (h, w)

def run_gradcam(
    model_path: str,
    image: np.ndarray,
    target_class: int = None,
    alpha: float = 0.5,
    img_size: int = 640,
):
    model = YOLO(model_path)
    pytorch_model = model.model
    pytorch_model.eval()

    target_layer = get_target_layer(model)
    gradcam = GradCAM(pytorch_model, target_layer)

    tensor, scale, orig_shape = preprocess_for_gradcam(image, img_size)

    try:
        cam = gradcam.generate(tensor, target_class=target_class)
    except Exception as e:
        print(f"GradCAM generation error: {e}")
        gradcam.remove_hooks()
        return None, None, None

    if cam is None:
        gradcam.remove_hooks()
        return None, None, None

    overlay, cam_resized = gradcam.overlay_heatmap(cam, image, alpha=alpha)
    gradcam.remove_hooks()

    return cam, overlay, cam_resized

def run_gradcam_multi_class(
    model_path: str,
    image: np.ndarray,
    num_classes: int,
    alpha: float = 0.5,
):
    results = {}
    for cls_id in range(num_classes):
        cam, overlay, cam_resized = run_gradcam(
            model_path, image,
            target_class=cls_id,
            alpha=alpha
        )
        if cam is not None:
            results[cls_id] = {
                "cam": cam,
                "overlay": overlay,
                "cam_resized": cam_resized,
            }
    return results