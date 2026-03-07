import streamlit as st
import torch
from ultralytics import YOLO
from pathlib import Path

st.title("Vision Inspection Portfolio")
st.success("Environment Ready!")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PyTorch", torch.__version__)
with col2:
    model_path = Path("models/best.onnx")
    st.metric("ONNX Model", "Ready ✅" if model_path.exists() else "Missing ❌")
with col3:
    data_path = Path("data/processed/bottle")
    st.metric("Dataset", "Ready ✅" if data_path.exists() else "Missing ❌")

st.header("Model Info")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1024 / 1024
    st.write(f"Model size: {size_mb:.1f} MB")
    st.write(f"Model path: {model_path.absolute()}")