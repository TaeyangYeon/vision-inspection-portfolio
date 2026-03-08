import streamlit as st
import subprocess
import threading
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

def get_available_categories():
    processed_dir = BASE_PATH / "data/processed"
    if not processed_dir.exists():
        return []
    return [d.name for d in processed_dir.iterdir() if d.is_dir()]

def get_latest_results(run_name: str):
    results_path = BASE_PATH / f"outputs/{run_name}/results.csv"
    if not results_path.exists():
        return None
    try:
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

def render_training_params():
    st.subheader("Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        categories = get_available_categories()
        category = st.selectbox("Dataset Category", categories if categories else ["bottle"])

        model_size = st.selectbox(
            "YOLOv8 Model Size",
            ["yolov8n", "yolov8s", "yolov8m"],
            help="n=nano (fastest), s=small, m=medium (most accurate)"
        )

        epochs = st.slider("Epochs", 10, 200, 100, step=10)
        batch_size = st.slider("Batch Size", 4, 32, 16, step=4)

    with col2:
        img_size = st.selectbox("Image Size", [416, 512, 640], index=2)
        patience = st.slider("Early Stopping Patience", 5, 50, 20)
        lr = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.005, 0.01, 0.05],
            value=0.01
        )
        run_name = st.text_input("Run Name", value=f"{category}_run")

    st.markdown("---")
    st.subheader("Augmentation Settings")

    col3, col4 = st.columns(2)
    with col3:
        mosaic = st.slider("Mosaic", 0.0, 1.0, 1.0)
        flipud = st.slider("Flip Up-Down", 0.0, 1.0, 0.3)
        fliplr = st.slider("Flip Left-Right", 0.0, 1.0, 0.5)
    with col4:
        hsv_h = st.slider("HSV Hue", 0.0, 0.1, 0.015)
        hsv_s = st.slider("HSV Saturation", 0.0, 1.0, 0.7)
        hsv_v = st.slider("HSV Value", 0.0, 1.0, 0.4)

    return {
        "category": category,
        "model_size": model_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "patience": patience,
        "lr": lr,
        "run_name": run_name,
        "mosaic": mosaic,
        "flipud": flipud,
        "fliplr": fliplr,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
    }

def render_train_tab():
    st.header("Training Monitor")

    params = render_training_params()

    st.markdown("---")

    col_start, col_stop, col_status = st.columns([1, 1, 3])

    with col_start:
        start_clicked = st.button("Start Training", type="primary")
    with col_stop:
        stop_clicked = st.button("Stop Training")
    with col_status:
        if "training_status" not in st.session_state:
            st.session_state.training_status = "idle"
        status = st.session_state.training_status
        if status == "idle":
            st.info("Status: Ready")
        elif status == "running":
            st.warning("Status: Training in progress...")
        elif status == "done":
            st.success("Status: Training complete")
        elif status == "stopped":
            st.error("Status: Training stopped")

    if start_clicked:
        st.session_state.training_status = "running"
        st.session_state.train_params = params

        yaml_path = BASE_PATH / f"data/processed/{params['category']}/dataset.yaml"
        output_dir = BASE_PATH / "outputs"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "-m", "ultralytics",
            "train",
            f"model={params['model_size']}.pt",
            f"data={yaml_path}",
            f"epochs={params['epochs']}",
            f"imgsz={params['img_size']}",
            f"batch={params['batch_size']}",
            f"patience={params['patience']}",
            f"lr0={params['lr']}",
            f"name={params['run_name']}",
            f"project={output_dir}",
            f"mosaic={params['mosaic']}",
            f"flipud={params['flipud']}",
            f"fliplr={params['fliplr']}",
            f"hsv_h={params['hsv_h']}",
            f"hsv_s={params['hsv_s']}",
            f"hsv_v={params['hsv_v']}",
            "save=True",
            "plots=True",
        ]

        st.session_state.train_cmd = " ".join(cmd)
        st.info("Training command prepared. In production, this runs on GPU server (Colab).")
        st.code(st.session_state.train_cmd)

    if stop_clicked:
        st.session_state.training_status = "stopped"
        st.rerun()

    st.markdown("---")
    render_results_section(params.get("run_name", ""))

def render_results_section(run_name: str):
    st.subheader("Training Results")

    results_options = []
    outputs_dir = BASE_PATH / "outputs"
    if outputs_dir.exists():
        results_options = [d.name for d in outputs_dir.iterdir() if d.is_dir()]

    if not results_options:
        st.info("No training results found. Run training first.")
        return

    selected_run = st.selectbox("Select Run", results_options)
    df = get_latest_results(selected_run)

    if df is None:
        st.warning(f"No results.csv found for run: {selected_run}")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "metrics/mAP50(B)" in df.columns:
            best_map = df["metrics/mAP50(B)"].max()
            st.metric("Best mAP50", f"{best_map:.4f}")
    with col2:
        if "metrics/precision(B)" in df.columns:
            best_prec = df["metrics/precision(B)"].max()
            st.metric("Best Precision", f"{best_prec:.4f}")
    with col3:
        if "metrics/recall(B)" in df.columns:
            best_rec = df["metrics/recall(B)"].max()
            st.metric("Best Recall", f"{best_rec:.4f}")
    with col4:
        st.metric("Total Epochs", len(df))

    tab1, tab2 = st.tabs(["Loss Curves", "mAP Curve"])

    with tab1:
        fig = go.Figure()
        if "train/box_loss" in df.columns:
            fig.add_trace(go.Scatter(y=df["train/box_loss"], name="Train Box Loss", mode="lines"))
        if "train/cls_loss" in df.columns:
            fig.add_trace(go.Scatter(y=df["train/cls_loss"], name="Train Cls Loss", mode="lines"))
        if "val/box_loss" in df.columns:
            fig.add_trace(go.Scatter(y=df["val/box_loss"], name="Val Box Loss", mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        if "metrics/mAP50(B)" in df.columns:
            fig2.add_trace(go.Scatter(y=df["metrics/mAP50(B)"], name="mAP50", mode="lines"))
        if "metrics/mAP50-95(B)" in df.columns:
            fig2.add_trace(go.Scatter(y=df["metrics/mAP50-95(B)"], name="mAP50-95", mode="lines", line=dict(dash="dash")))
        fig2.update_layout(title="mAP Curve", xaxis_title="Epoch", yaxis_title="mAP")
        st.plotly_chart(fig2, use_container_width=True)