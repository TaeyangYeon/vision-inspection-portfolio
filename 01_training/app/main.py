import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from pages.data_tab import render_data_tab
from pages.train_tab import render_train_tab
from pages.eval_tab import render_eval_tab
from pages.export_tab import render_export_tab

st.set_page_config(
    page_title="Vision Model Trainer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Vision Model Trainer")

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

tab = st.sidebar.radio(
    "Select Tab",
    ["Data", "Train", "Eval", "Export"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Status")

base_path = Path(__file__).parent.parent
model_exists = (base_path / "models/best.onnx").exists()
data_exists = (base_path / "data/processed/bottle").exists()

st.sidebar.markdown(f"**ONNX Model:** {'Ready' if model_exists else 'Missing'}")
st.sidebar.markdown(f"**Dataset:** {'Ready' if data_exists else 'Missing'}")
st.sidebar.markdown("---")
st.sidebar.caption("Vision Inspection Portfolio v1.0")

if tab == "Data":
    render_data_tab()
elif tab == "Train":
    render_train_tab()
elif tab == "Eval":
    render_eval_tab()
elif tab == "Export":
    render_export_tab()