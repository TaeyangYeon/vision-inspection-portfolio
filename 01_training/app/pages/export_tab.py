import streamlit as st
import numpy as np
import cv2
import time
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import yaml

BASE_PATH = Path(__file__).parent.parent.parent

def get_available_runs():
    outputs_dir = BASE_PATH / "outputs"
    if not outputs_dir.exists():
        return []
    return [d.name for d in outputs_dir.iterdir() if d.is_dir()]

def load_class_names(category: str):
    yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {})

def preprocess_for_compare(image: np.ndarray, img_size: int = 640):
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

def run_onnx_inference(model_path: str, tensor: np.ndarray):
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    output = session.run(None, {input_name: tensor})
    elapsed = (time.perf_counter() - start) * 1000
    return output[0], elapsed

def run_pt_inference(model_path: str, tensor: np.ndarray):
    import torch
    from ultralytics import YOLO
    model = YOLO(model_path)
    img_tensor = torch.from_numpy(tensor)
    start = time.perf_counter()
    with torch.no_grad():
        result = model.predict(source=img_tensor, verbose=False, imgsz=640)
    elapsed = (time.perf_counter() - start) * 1000
    boxes = result[0].boxes
    if boxes is not None and len(boxes) > 0:
        output = boxes.data.cpu().numpy()
    else:
        output = np.zeros((0, 6))
    return output, elapsed

def parse_onnx_output(output: np.ndarray, conf_thresh: float, scale: float, orig_shape: tuple):
    predictions = output[0]
    if predictions.ndim == 3:
        predictions = predictions[0]

    boxes, scores, class_ids = [], [], []
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
    return boxes, scores, class_ids

def draw_boxes_on_image(image: np.ndarray, boxes, scores, class_ids, class_names, color):
    result = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls_id, str(cls_id))}: {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result

def render_model_info():
    st.subheader("Model Files")

    col1, col2 = st.columns(2)

    onnx_path = BASE_PATH / "models/best.onnx"
    with col1:
        st.markdown("**ONNX Model**")
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            st.success(f"Ready - {size_mb:.1f} MB")
            st.caption(str(onnx_path))
        else:
            st.error("Not found")

    runs = get_available_runs()
    with col2:
        st.markdown("**PyTorch Model (.pt)**")
        pt_found = False
        for run in runs:
            pt_path = BASE_PATH / f"outputs/{run}/weights/best.pt"
            if pt_path.exists():
                size_mb = pt_path.stat().st_size / 1024 / 1024
                st.success(f"Ready - {size_mb:.1f} MB ({run})")
                st.caption(str(pt_path))
                pt_found = True
                break
        if not pt_found:
            st.warning("Not found locally (trained on Colab)")

def render_speed_benchmark():
    st.subheader("Speed Benchmark - ONNX vs PyTorch")
    st.caption("Compare inference speed between ONNX Runtime and PyTorch")

    onnx_path = BASE_PATH / "models/best.onnx"
    if not onnx_path.exists():
        st.error("ONNX model not found.")
        return

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not categories:
        st.error("No datasets found.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        category = st.selectbox("Category", categories, key="export_category")
        n_runs = st.slider("Benchmark Iterations", 5, 50, 10)
        conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.25, key="export_conf")
        run_benchmark = st.button("Run Speed Benchmark", type="primary")

    with col2:
        if run_benchmark:
            img_dir = BASE_PATH / f"data/processed/{category}/images/val"
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

            if not images:
                st.warning("No val images found.")
                return

            import random
            sample_imgs = random.sample(images, min(n_runs, len(images)))

            onnx_times = []
            progress = st.progress(0)

            for idx, img_path in enumerate(sample_imgs):
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor, scale, orig_shape = preprocess_for_compare(img_rgb)
                _, elapsed = run_onnx_inference(str(onnx_path), tensor)
                onnx_times.append(elapsed)
                progress.progress((idx + 1) / len(sample_imgs))

            avg_onnx = np.mean(onnx_times)
            fps_onnx = 1000 / avg_onnx

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Avg Inference Time", f"{avg_onnx:.1f} ms")
            with mc2:
                st.metric("Estimated FPS", f"{fps_onnx:.1f}")
            with mc3:
                st.metric("Iterations", len(onnx_times))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=onnx_times,
                mode="lines+markers",
                name="ONNX Runtime",
                line=dict(color="#1f77b4", width=2)
            ))
            fig.add_hline(
                y=avg_onnx,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {avg_onnx:.1f}ms"
            )
            fig.update_layout(
                title="ONNX Inference Time per Image",
                xaxis_title="Iteration",
                yaxis_title="Time (ms)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click Run Speed Benchmark to measure inference speed.")

def render_output_comparison():
    st.subheader("PT vs ONNX Output Comparison")
    st.caption("Compare detection results between PyTorch and ONNX model on same image")

    onnx_path = BASE_PATH / "models/best.onnx"
    if not onnx_path.exists():
        st.error("ONNX model not found.")
        return

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    col1, col2 = st.columns([1, 3])

    with col1:
        category = st.selectbox("Category", categories, key="compare_category")
        conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.25, key="compare_conf")
        class_names = load_class_names(category)

        img_dir = BASE_PATH / f"data/processed/{category}/images/val"
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not images:
            st.warning("No val images found.")
            return

        img_names = [p.name for p in images]
        selected_name = st.selectbox("Select Image", img_names, key="compare_img")
        selected_path = img_dir / selected_name
        run_compare = st.button("Run Comparison", type="primary")

    with col2:
        if run_compare:
            img_bgr = cv2.imread(str(selected_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor, scale, orig_shape = preprocess_for_compare(img_rgb)

            with st.spinner("Running ONNX inference..."):
                onnx_output, onnx_time = run_onnx_inference(str(onnx_path), tensor)
                onnx_boxes, onnx_scores, onnx_cls = parse_onnx_output(
                    onnx_output, conf_thresh, scale, orig_shape
                )

            onnx_img = draw_boxes_on_image(
                img_rgb.copy(), onnx_boxes, onnx_scores, onnx_cls,
                class_names, (50, 200, 50)
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"ONNX Runtime ({onnx_time:.1f}ms) - {len(onnx_boxes)} detections")
                st.image(onnx_img, use_container_width=True)
            with col_b:
                st.caption("PyTorch (.pt) - not available locally (trained on Colab)")
                st.image(img_rgb, use_container_width=True, caption="Original image")

            st.markdown("---")
            st.markdown("**ONNX Detection Results**")
            if onnx_boxes:
                rows = []
                for box, score, cls_id in zip(onnx_boxes, onnx_scores, onnx_cls):
                    rows.append({
                        "Class": class_names.get(cls_id, str(cls_id)),
                        "Confidence": round(score, 4),
                        "BBox": str(box),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No detections found.")
        else:
            if selected_path.exists():
                img = cv2.imread(str(selected_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True, caption=selected_name)

def render_export_tab():
    st.header("ONNX Export")

    render_model_info()
    st.markdown("---")
    render_speed_benchmark()
    st.markdown("---")
    render_output_comparison()