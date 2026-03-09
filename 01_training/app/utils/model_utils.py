import numpy as np
import cv2
from pathlib import Path

def preprocess_image(image: np.ndarray, img_size: int = 640):
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, (h, w)

def postprocess_output(output: np.ndarray, conf_thresh: float, iou_thresh: float, orig_shape: tuple, scale: float, img_size: int = 640):
    predictions = output[0]
    if predictions.ndim == 3:
        predictions = predictions[0]

    boxes = []
    scores = []
    class_ids = []

    num_classes = predictions.shape[0] - 4

    for i in range(predictions.shape[1]):
        pred = predictions[:, i]
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        class_scores = pred[4:4 + num_classes]
        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])

        if conf < conf_thresh:
            continue

        x1 = int((cx - bw / 2) / scale)
        y1 = int((cy - bh / 2) / scale)
        x2 = int((cx + bw / 2) / scale)
        y2 = int((cy + bh / 2) / scale)

        x1 = max(0, min(x1, orig_shape[1]))
        y1 = max(0, min(y1, orig_shape[0]))
        x2 = max(0, min(x2, orig_shape[1]))
        y2 = max(0, min(y2, orig_shape[0]))

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
        scores,
        conf_thresh,
        iou_thresh
    )

    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]

def draw_detections(image: np.ndarray, boxes: list, scores: list, class_ids: list, class_names: dict):
    colors = [
        (255, 50, 50),
        (50, 200, 50),
        (50, 50, 255),
        (255, 200, 50),
        (200, 50, 255),
    ]
    result = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = colors[cls_id % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls_id, str(cls_id))}: {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return result