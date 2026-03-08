import streamlit as st
import cv2
import numpy as np
import yaml
from pathlib import Path
import random
import plotly.express as px
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.parent

def load_dataset_info(category: str):
    yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def get_image_list(category: str, split: str):
    img_dir = BASE_PATH / f"data/processed/{category}/images/{split}"
    if not img_dir.exists():
        return []
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

def draw_labels(image, label_path, class_names):
    h, w = image.shape[:2]
    if not label_path.exists():
        return image
    with open(label_path, "r") as f:
        lines = f.readlines()
    colors = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 50, 255),
        (255, 255, 50),
        (255, 50, 255),
    ]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names.get(cls_id, str(cls_id))
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def render_class_distribution(category: str, split: str, class_names: dict):
    label_dir = BASE_PATH / f"data/processed/{category}/labels/{split}"
    if not label_dir.exists():
        return

    class_counts = {v: 0 for v in class_names.values()}
    empty_count = 0

    for label_file in label_dir.glob("*.txt"):
        content = label_file.read_text().strip()
        if not content:
            empty_count += 1
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                cls_name = class_names.get(cls_id, str(cls_id))
                if cls_name in class_counts:
                    class_counts[cls_name] += 1

    st.subheader("Class Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        df = pd.DataFrame({
            "Class": list(class_counts.keys()),
            "Count": list(class_counts.values())
        })
        fig = px.bar(df, x="Class", y="Count",
                     title=f"{category} - {split} defect counts",
                     color="Class")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        total_defect = sum(class_counts.values())
        st.metric("Total defect instances", total_defect)
        st.metric("Clean images", empty_count)
        st.metric("Defect ratio", f"{total_defect/(total_defect+empty_count)*100:.1f}%" if (total_defect+empty_count) > 0 else "N/A")

def render_data_tab():
    st.header("Data Management")

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not categories:
        st.error("No processed datasets found. Run convert_to_yolo.py first.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")
        category = st.selectbox("Category", categories)
        split = st.selectbox("Split", ["train", "val"])

        dataset_info = load_dataset_info(category)
        class_names = {}
        if dataset_info and "names" in dataset_info:
            class_names = dataset_info["names"]

        show_labels = st.checkbox("Show BBox Labels", value=True)

        images = get_image_list(category, split)
        st.markdown(f"**Total images:** {len(images)}")

        if images:
            img_names = [p.name for p in images]
            selected_name = st.selectbox("Select Image", img_names)
            selected_path = BASE_PATH / f"data/processed/{category}/images/{split}/{selected_name}"
        else:
            st.warning("No images found.")
            return

        if st.button("Random Image"):
            selected_path = random.choice(images)
            st.rerun()

    with col2:
        st.subheader("Image Viewer")
        if selected_path.exists():
            img = cv2.imread(str(selected_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if show_labels:
                label_path = BASE_PATH / f"data/processed/{category}/labels/{split}/{selected_path.stem}.txt"
                img = draw_labels(img.copy(), label_path, class_names)
                if label_path.exists() and label_path.stat().st_size > 0:
                    st.success("Defect detected - labels shown")
                else:
                    st.info("No defects - clean image")

            st.image(img, use_column_width=True, caption=selected_path.name)
        else:
            st.error("Image not found.")

    st.markdown("---")
    render_class_distribution(category, split, class_names)