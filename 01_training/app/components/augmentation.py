import cv2
import numpy as np
import random

def apply_augmentations(image: np.ndarray, config: dict) -> dict:
    results = {}
    results["Original"] = image.copy()

    if config.get("horizontal_flip"):
        results["Horizontal Flip"] = cv2.flip(image, 1)

    if config.get("vertical_flip"):
        results["Vertical Flip"] = cv2.flip(image, 0)

    if config.get("brightness"):
        factor = config.get("brightness_factor", 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        results["Brightness"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if config.get("gaussian_noise"):
        noise = np.random.normal(0, 25, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        results["Gaussian Noise"] = noisy

    if config.get("rotation"):
        angle = config.get("rotation_angle", 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        results["Rotation"] = cv2.warpAffine(image, M, (w, h))

    if config.get("blur"):
        kernel = config.get("blur_kernel", 5)
        if kernel % 2 == 0:
            kernel += 1
        results["Blur"] = cv2.GaussianBlur(image, (kernel, kernel), 0)

    if config.get("hsv_shift"):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + config.get("hue_shift", 20)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * config.get("sat_factor", 1.3), 0, 255)
        results["HSV Shift"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if config.get("mosaic"):
        h, w = image.shape[:2]
        mosaic = image.copy()
        cx, cy = w // 2, h // 2
        quadrants = [
            (image[:cy, :cx], 0, 0),
            (cv2.flip(image[:cy, cx:], 1), 0, cx),
            (cv2.flip(image[cy:, :cx], 0), cy, 0),
            (cv2.flip(image[cy:, cx:], -1), cy, cx),
        ]
        for quad, y, x in quadrants:
            qh, qw = quad.shape[:2]
            mosaic[y:y+qh, x:x+qw] = quad
        results["Mosaic"] = mosaic

    return results