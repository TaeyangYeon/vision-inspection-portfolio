import streamlit as st
import numpy as np
import cv2
import torch
from pathlib import Path
import sys

BASE_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_PATH / "gradcam"))

def render_gradcam_tab():
    st.header("GradCAM Visualization")
    st.caption("Custom PyTorch hook implementation - no external library")

    pt_path = BASE_PATH / "models/best.pt"
    if not pt_path.exists():
        st.error("best.pt not found at models/best.pt")
        st.info("Download from Google Drive: vision_portfolio/runs/bottle_full/weights/best.pt")
        return

    categories = []
    processed_dir = BASE_PATH / "data/processed"
    if processed_dir.exists():
        categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not categories:
        st.error("No datasets found.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")
        category = st.selectbox("Category", categories, key="gcam_category")
        split = st.selectbox("Split", ["val", "train"], key="gcam_split")

        import yaml
        yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
        class_names = {}
        num_classes = 3
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
                class_names = yaml_data.get("names", {})
                num_classes = yaml_data.get("nc", 3)

        target_class = st.selectbox(
            "Target Class",
            options=list(class_names.keys()),
            format_func=lambda x: f"{x}: {class_names.get(x, str(x))}",
            key="gcam_class"
        )

        alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5, key="gcam_alpha")

        img_dir = BASE_PATH / f"data/processed/{category}/images/{split}"
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        defect_images = [p for p in images if not p.stem.isdigit()]
        all_images = defect_images + [p for p in images if p not in defect_images]

        if not all_images:
            st.warning("No images found.")
            return

        img_names = [p.name for p in all_images]
        selected_name = st.selectbox("Select Image", img_names, key="gcam_img")
        selected_path = img_dir / selected_name

        run_gradcam = st.button("Generate GradCAM", type="primary")

        st.markdown("---")
        mode = st.radio(
            "Mode",
            ["Single Class", "All Classes"],
            key="gcam_mode"
        )
        st.markdown("---")
        st.caption("How it works:")
        st.caption("1. Register forward/backward hooks")
        st.caption("2. Forward pass extracts activations")
        st.caption("3. Backward pass extracts gradients")
        st.caption("4. Weighted sum = class activation map")

    with col2:
        if selected_path.exists():
            img_bgr = cv2.imread(str(selected_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if run_gradcam:
                with st.spinner("Running GradCAM (PyTorch hooks)..."):
                    try:
                        from gradcam_yolo import run_gradcam as _run_gradcam
                        from gradcam_multiclass import run_multiclass_comparison

                        if mode == "Single Class":
                            cam, overlay, cam_resized = _run_gradcam(
                                str(pt_path),
                                img_rgb,
                                target_class=int(target_class),
                                alpha=alpha,
                            )

                            if cam is None:
                                st.error("GradCAM generation failed.")
                            else:
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.caption("Original")
                                    st.image(img_rgb, use_container_width=True)
                                with col_b:
                                    st.caption("GradCAM Heatmap")
                                    heatmap = cv2.applyColorMap(
                                        (cam_resized * 255).astype(np.uint8),
                                        cv2.COLORMAP_JET
                                    )
                                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                    st.image(heatmap_rgb, use_container_width=True)
                                with col_c:
                                    st.caption("Overlay")
                                    st.image(overlay, use_container_width=True)

                                st.success(
                                    f"GradCAM generated for class: "
                                    f"{class_names.get(int(target_class), str(target_class))}"
                                )
                                st.markdown("---")
                                st.subheader("CAM Statistics")
                                cam_stats = st.columns(3)
                                with cam_stats[0]:
                                    st.metric("CAM Min", f"{cam.min():.4f}")
                                with cam_stats[1]:
                                    st.metric("CAM Max", f"{cam.max():.4f}")
                                with cam_stats[2]:
                                    st.metric("CAM Mean", f"{cam.mean():.4f}")

                        else:
                            results = run_multiclass_comparison(
                                str(pt_path), img_rgb, class_names,
                                output_dir=BASE_PATH / "outputs/gradcam_multiclass/streamlit",
                                alpha=alpha,
                            )

                            if not results:
                                st.error("Multi-class GradCAM failed.")
                            else:
                                st.success(f"Generated GradCAM for {len(results)} classes")
                                st.markdown("---")

                                orig_col, *class_cols = st.columns(len(results) + 1)
                                with orig_col:
                                    st.caption("Original")
                                    st.image(img_rgb, use_container_width=True)

                                for col_widget, (cls_id, data) in zip(class_cols, results.items()):
                                    with col_widget:
                                        st.caption(f"{data['name']}")
                                        st.image(data["overlay"], use_container_width=True)
                                        st.caption(f"mean: {data['mean']:.3f}")

                                st.markdown("---")
                                st.subheader("Activation Comparison")
                                import plotly.graph_objects as go
                                fig = go.Figure(go.Bar(
                                    x=[data["name"] for data in results.values()],
                                    y=[data["mean"] for data in results.values()],
                                    marker_color="#0078D4",
                                    text=[f"{data['mean']:.3f}" for data in results.values()],
                                    textposition="outside",
                                ))
                                fig.update_layout(
                                    title="Mean Activation per Class",
                                    xaxis_title="Class",
                                    yaxis_title="Mean CAM Activation",
                                    yaxis=dict(range=[0, 1]),
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
            else:
                st.image(img_rgb, use_container_width=True, caption=selected_name)
                st.info("Select settings and click Generate GradCAM.")