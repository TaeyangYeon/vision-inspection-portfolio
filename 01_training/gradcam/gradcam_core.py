import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.gradients = []
        self.activations = []

        self.model.eval()
        output = self.model(input_tensor)

        if isinstance(output, (list, tuple)):
            output = output[0]

        if output.dim() == 3:
            pred = output[0]
            class_scores = pred[4:, :]
            scores_sum = class_scores.sum(dim=1)
            if target_class is None:
                target_class = int(scores_sum.argmax())
            score = scores_sum[target_class]
        else:
            if target_class is None:
                target_class = int(output.argmax())
            score = output[0, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if not self.gradients or not self.activations:
            return None

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, cam: np.ndarray, image: np.ndarray, alpha: float = 0.5):
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
        return overlay, cam_resized

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()