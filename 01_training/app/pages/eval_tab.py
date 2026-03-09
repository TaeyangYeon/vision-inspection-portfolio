import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import onnxruntime as ort
import yaml
from utils.model_utils import preprocess_image, postprocess_output, draw_detections
from utils.fpfn_utils import yolo_to_pixel, classify_detections, draw_fpfn
import cv2

BASE_PATH = Path(__file__).parent.parent.parent

def render_fpfn_viewer():
    st.subheader("FP / FN Case Analysis")
    st.caption("TP=green / FP=red (false alarm) / FN=blue (missed defect)")

    model_path = BASE_PATH / "models/best.onnx"
    if not model_path.exists():
        st.error("ONNX model not found.")
        return

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not categories:
        st.error("No processed datasets found.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        category = st.selectbox("Category", categories, key="fpfn_category")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, key="fpfn_conf")
        iou_thresh = st.slider("IoU Threshold", 0.1, 0.9, 0.5, key="fpfn_iou")
        filter_type = st.selectbox(
            "Filter by Case",
            ["All", "Has FP", "Has FN", "Has TP only"],
            key="fpfn_filter"
        )
        run_analysis = st.button("Run FP/FN Analysis", type="primary")

    with col2:
        if run_analysis:
            import onnxruntime as ort
            from utils.model_utils import preprocess_image, postprocess_output

            yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
            class_names = {}
            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    class_names = yaml_data.get("names", {})

            val_img_dir = BASE_PATH / f"data/processed/{category}/images/val"
            val_lbl_dir = BASE_PATH / f"data/processed/{category}/labels/val"
            images = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))

            if not images:
                st.warning("No val images found.")
                return

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            results = []
            progress = st.progress(0)

            for idx, img_path in enumerate(images):
                img_bgr = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
                gt_boxes = []
                gt_cls = []
                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    for line in lbl_path.read_text().strip().splitlines():
                        cls_id, box = yolo_to_pixel(line, w, h)
                        gt_boxes.append(box)
                        gt_cls.append(cls_id)

                tensor, scale, orig_shape = preprocess_image(img_rgb)
                output = session.run(None, {input_name: tensor})
                pred_boxes, pred_scores, pred_cls = postprocess_output(
                    output[0], conf_thresh, iou_thresh, orig_shape, scale
                )

                tp, fp, fn = classify_detections(
                    pred_boxes, pred_scores, pred_cls,
                    gt_boxes, gt_cls, iou_thresh
                )

                results.append({
                    "img_path": img_path,
                    "img_rgb": img_rgb,
                    "tp": tp, "fp": fp, "fn": fn,
                    "gt_boxes": gt_boxes, "gt_cls": gt_cls,
                })

                progress.progress((idx + 1) / len(images))

            if filter_type == "Has FP":
                results = [r for r in results if r["fp"]]
            elif filter_type == "Has FN":
                results = [r for r in results if r["fn"]]
            elif filter_type == "Has TP only":
                results = [r for r in results if r["tp"] and not r["fp"] and not r["fn"]]

            total_tp = sum(len(r["tp"]) for r in results)
            total_fp = sum(len(r["fp"]) for r in results)
            total_fn = sum(len(r["fn"]) for r in results)

            st.markdown("---")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Total TP", total_tp, help="Correct detections")
            with mc2:
                st.metric("Total FP", total_fp, help="False alarms")
            with mc3:
                st.metric("Total FN", total_fn, help="Missed defects")

            st.markdown("---")
            st.write(f"Showing {len(results)} images after filter: {filter_type}")

            for r in results[:12]:
                annotated = draw_fpfn(
                    r["img_rgb"], r["tp"], r["fp"], r["fn"], class_names
                )
                caption = f"{r['img_path'].name} | TP:{len(r['tp'])} FP:{len(r['fp'])} FN:{len(r['fn'])}"
                st.image(annotated, caption=caption, use_container_width=True)
        else:
            st.info("Set parameters and click Run FP/FN Analysis.")

def render_sample_inference(run_name: str):
    st.subheader("Sample Image Inference Viewer")
    st.caption("Run real ONNX model inference on dataset images")

    model_path = BASE_PATH / "models/best.onnx"
    if not model_path.exists():
        st.error("ONNX model not found at models/best.onnx")
        return

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not categories:
        st.error("No processed datasets found.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        category = st.selectbox("Category", categories, key="infer_category")
        split = st.selectbox("Split", ["val", "train"], key="infer_split")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, key="infer_conf")
        iou_thresh = st.slider("IoU Threshold", 0.1, 0.9, 0.45, key="infer_iou")

        yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
        class_names = {}
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                class_names = yaml_data.get("names", {})

        img_dir = BASE_PATH / f"data/processed/{category}/images/{split}"
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        if not images:
            st.warning("No images found.")
            return

        img_names = [p.name for p in images]
        selected_name = st.selectbox("Select Image", img_names, key="infer_img")
        selected_path = img_dir / selected_name

        run_inference = st.button("Run Inference", type="primary")

    with col2:
        if selected_path.exists():
            img_bgr = cv2.imread(str(selected_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            label_path = BASE_PATH / f"data/processed/{category}/labels/{split}/{selected_path.stem}.txt"
            has_label = label_path.exists() and label_path.stat().st_size > 0

            if run_inference:
                with st.spinner("Running ONNX inference..."):
                    session = ort.InferenceSession(str(model_path))
                    input_name = session.get_inputs()[0].name

                    tensor, scale, orig_shape = preprocess_image(img_rgb)
                    output = session.run(None, {input_name: tensor})

                    boxes, scores, class_ids = postprocess_output(
                        output[0], conf_thresh, iou_thresh, orig_shape, scale
                    )

                    result_img = draw_detections(img_rgb.copy(), boxes, scores, class_ids, class_names)

                col_orig, col_result = st.columns(2)
                with col_orig:
                    st.caption("Original")
                    st.image(img_rgb, use_container_width=True)
                with col_result:
                    st.caption(f"ONNX Inference Result ({len(boxes)} detections)")
                    st.image(result_img, use_container_width=True)

                if boxes:
                    st.success(f"NG - {len(boxes)} defect(s) detected")
                    det_data = []
                    for box, score, cls_id in zip(boxes, scores, class_ids):
                        det_data.append({
                            "Class": class_names.get(cls_id, str(cls_id)),
                            "Confidence": round(score, 4),
                            "BBox": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(det_data), use_container_width=True)
                else:
                    if has_label:
                        st.warning("OK (model missed defect - possible FN)")
                    else:
                        st.success("OK - No defects detected (clean image)")
            else:
                st.image(img_rgb, use_container_width=True, caption=selected_name)
                if has_label:
                    st.info("This image has defect labels. Click Run Inference to detect.")
                else:
                    st.info("Clean image (no defects). Click Run Inference to verify.")

def load_eval_results(run_name: str):
    path = BASE_PATH / f"outputs/{run_name}/eval_results.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def load_pr_data(run_name: str):
    path = BASE_PATH / f"outputs/{run_name}/pr_data.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def render_confusion_matrix(eval_data: dict):
    st.subheader("Confusion Matrix")

    cm = np.array(eval_data["confusion_matrix"])
    class_names = eval_data["class_names"]

    cm_normalized = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums

    text = [[f"{cm[i][j]}<br>({cm_normalized[i][j]:.2f})"
             for j in range(len(class_names))]
            for i in range(len(class_names))]

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        title="Confusion Matrix (normalized)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Each cell shows: count (normalized ratio). Diagonal = correct predictions.")

def render_per_class_metrics(eval_data: dict):
    st.subheader("Per-Class Metrics")

    per_class = eval_data.get("per_class", {})
    if not per_class:
        st.warning("No per-class data available.")
        return

    rows = []
    for cls_name, vals in per_class.items():
        rows.append({
            "Class": cls_name,
            "AP50": round(vals.get("ap50", 0), 4),
            "Precision": round(vals.get("precision", 0), 4),
            "Recall": round(vals.get("recall", 0), 4),
            "F1": round(2 * vals.get("precision", 0) * vals.get("recall", 0) /
                       (vals.get("precision", 0) + vals.get("recall", 0) + 1e-9), 4),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    fig = go.Figure()
    metrics_to_plot = ["AP50", "Precision", "Recall", "F1"]
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Class"],
            y=df[metric],
        ))

    fig.update_layout(
        barmode="group",
        title="Per-Class Metrics Comparison",
        xaxis_title="Class",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

def render_pr_curve(pr_data: dict):
    st.subheader("PR Curve")

    class_names = pr_data.get("class_names", [])
    px_vals = pr_data.get("px", [])
    py_vals = pr_data.get("py", [])

    if px_vals and py_vals:
        py_array = np.array(py_vals)
        px_array = np.array(px_vals)

        fig = go.Figure()
        colors = px.colors.qualitative.Set2

        for i, cls_name in enumerate(class_names):
            if i < py_array.shape[0]:
                fig.add_trace(go.Scatter(
                    x=px_array,
                    y=py_array[i],
                    mode="lines",
                    name=cls_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            legend_title="Class",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        precision = pr_data.get("precision", [])
        recall = pr_data.get("recall", [])

        if precision and recall:
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, cls_name in enumerate(class_names):
                if i < len(precision):
                    fig.add_trace(go.Scatter(
                        x=[0, recall[i], 1],
                        y=[1, precision[i], 0],
                        mode="lines+markers",
                        name=f"{cls_name} (P:{precision[i]:.2f} R:{recall[i]:.2f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            fig.update_layout(
                title="Precision-Recall Summary",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

def render_f1_curve(pr_data: dict):
    st.subheader("F1 Score by Class")

    class_names = pr_data.get("class_names", [])
    precision = pr_data.get("precision", [])
    recall = pr_data.get("recall", [])

    if not precision or not recall:
        st.warning("No F1 data available.")
        return

    f1_scores = []
    for p, r in zip(precision, recall):
        f1 = 2 * p * r / (p + r + 1e-9)
        f1_scores.append(round(f1, 4))

    df = pd.DataFrame({
        "Class": class_names[:len(f1_scores)],
        "F1 Score": f1_scores,
        "Precision": [round(p, 4) for p in precision[:len(f1_scores)]],
        "Recall": [round(r, 4) for r in recall[:len(f1_scores)]],
    })

    fig = px.bar(
        df, x="Class", y="F1 Score",
        color="F1 Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="F1 Score per Class",
        text="F1 Score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 1.1]))
    st.plotly_chart(fig, use_container_width=True)

def render_eval_tab():
    st.header("Evaluation Results")

    outputs_dir = BASE_PATH / "outputs"
    run_options = []
    if outputs_dir.exists():
        run_options = [d.name for d in outputs_dir.iterdir() if d.is_dir()]

    if not run_options:
        st.warning("No training runs found. Complete training first.")
        return

    selected_run = st.selectbox("Select Training Run", run_options)

    eval_data = load_eval_results(selected_run)
    pr_data = load_pr_data(selected_run)

    if eval_data is None and pr_data is None:
        st.error("No evaluation data found. Download eval_results.json and pr_data.json from Colab.")
        st.code("Expected path: outputs/" + selected_run + "/eval_results.json")
        return

    if eval_data:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("mAP50", f"{eval_data.get('mAP50', 0):.4f}")
        with col2:
            st.metric("mAP50-95", f"{eval_data.get('mAP50_95', 0):.4f}")
        with col3:
            st.metric("Precision", f"{eval_data.get('precision', 0):.4f}")
        with col4:
            st.metric("Recall", f"{eval_data.get('recall', 0):.4f}")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Per-Class Metrics", "PR + F1 Curve"])

        with tab1:
            render_confusion_matrix(eval_data)

        with tab2:
            render_per_class_metrics(eval_data)

        with tab3:
            if pr_data:
                render_pr_curve(pr_data)
                st.markdown("---")
                render_f1_curve(pr_data)
            else:
                st.warning("pr_data.json not found. Download from Colab.")
    else:
        st.error("eval_results.json not found.")

    st.markdown("---")
    render_sample_inference(selected_run)

    st.markdown("---")
    render_fpfn_viewer()