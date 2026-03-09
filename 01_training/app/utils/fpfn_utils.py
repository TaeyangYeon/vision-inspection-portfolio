import numpy as np
import cv2
from pathlib import Path
import yaml

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def yolo_to_pixel(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    cls_id = int(parts[0])
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return cls_id, [x1, y1, x2, y2]

def classify_detections(pred_boxes, pred_scores, pred_cls,
                         gt_boxes, gt_cls, iou_thresh=0.5):
    tp_list = []
    fp_list = []
    fn_list = []

    matched_gt = set()

    for i, (pb, ps, pc) in enumerate(zip(pred_boxes, pred_scores, pred_cls)):
        best_iou = 0
        best_gt_idx = -1
        for j, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
            if j in matched_gt:
                continue
            if pc != gc:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp_list.append((pb, ps, pc, best_iou))
            matched_gt.add(best_gt_idx)
        else:
            fp_list.append((pb, ps, pc, best_iou))

    for j, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
        if j not in matched_gt:
            fn_list.append((gb, gc))

    return tp_list, fp_list, fn_list

def draw_fpfn(image: np.ndarray, tp_list, fp_list, fn_list, class_names: dict):
    result = image.copy()

    for box, score, cls_id, iou in tp_list:
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), (50, 200, 50), 2)
        label = f"TP {class_names.get(cls_id, str(cls_id))} {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 200, 50), 2)

    for box, score, cls_id, iou in fp_list:
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 50, 50), 2)
        label = f"FP {class_names.get(cls_id, str(cls_id))} {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 50), 2)

    for box, cls_id in fn_list:
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), (50, 50, 255), 2)
        label = f"FN {class_names.get(cls_id, str(cls_id))}"
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)

    return result