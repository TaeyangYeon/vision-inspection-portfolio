import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

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