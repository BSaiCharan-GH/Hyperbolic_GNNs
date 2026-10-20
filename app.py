"""Streamlit interface for the original Lorentz-model forensic analytics project.

No synthetic transactions or invented model metrics are generated. The explorer is
activated by adding the real predictions.csv exported by export_predictions.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS_PATH = ARTIFACTS / "predictions.csv"
METRICS_PATH = ARTIFACTS / "deployment_metrics.json"
HISTORY_PATH = ARTIFACTS / "training_history.json"
REQUIRED_COLUMNS = {"tx_id", "timestep", "actual_label", "illicit_score"}

st.set_page_config(
    page_title="Hyperbolic GNN | Forensic Analytics",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { background: radial-gradient(ellipse at 82% 0%, rgba(49,100,160,.22), transparent 38%), #08111f; color:#eaf1ff; }
    [data-testid="stSidebar"] { background:#0c1728; border-right:1px solid rgba(160,183,220,.16); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:#f4f7ff; }
    p,label { color:#c6d2e8; }
    .hero { padding:1.8rem; border:1px solid rgba(104,224,207,.25); border-radius:22px; background:linear-gradient(120deg,#142b45,#0d192b); margin-bottom:1rem; }
    .eyebrow { color:#68e0cf; font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.7rem; }
    .hero h1 { font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; margin:0 0 .8rem; }
    .hero p,.subtle { color:#b8c7df; line-height:1.7; }
    .pill { display:inline-block; border:1px solid rgba(104,224,207,.35); color:#8af0df; background:rgba(104,224,207,.08); border-radius:99px; padding:.28rem .68rem; font-size:.76rem; margin:.25rem .25rem 0 0; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#132137,#0d1829); border:1px solid rgba(160,183,220,.16); padding:.85rem; border-radius:14px; }
    div[data-testid="stMetricValue"] { color:#f4f7ff; font-family:'Space Grotesk',sans-serif; }
    .stButton > button { border-radius:10px; min-height:2.5rem; font-weight:600; }
    hr { border-color:rgba(160,183,220,.16); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_metrics() -> dict:
    data = read_json(METRICS_PATH)
    return data if isinstance(data, dict) else {}


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("Prediction export is missing columns: " + ", ".join(sorted(missing)))
    df["tx_id"] = df["tx_id"].astype(str)
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
    df["illicit_score"] = pd.to_numeric(df["illicit_score"], errors="coerce")
    df = df.dropna(subset=["timestep", "illicit_score"])
    if df.empty:
        raise ValueError("The prediction export contains no valid rows")
    if not np.isfinite(df["illicit_score"].to_numpy()).all():
        raise ValueError("Scores must be finite numeric values")
    if ((df["illicit_score"] < 0) | (df["illicit_score"] > 1)).any():
        raise ValueError("Scores must be between 0 and 1")
    df["timestep"] = df["timestep"].astype(int)
    df["actual_label"] = df["actual_label"].astype(str).str.strip().str.title()
    df["actual_label"] = df["actual_label"].replace({"1": "Illicit", "2": "Licit", "Unknown": "Unknown"})
    bad_labels = set(df["actual_label"].unique()) - {"Illicit", "Licit", "Unknown"}
    if bad_labels:
        raise ValueError("Unrecognized actual labels: " + ", ".join(sorted(bad_labels)))
    for col in ("degree", "embedding_x", "embedding_y"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def calculate_metrics(df: pd.DataFrame, threshold: float) -> dict | None:
    test = df[df["timestep"].between(42, 49) & df["actual_label"].isin(["Illicit", "Licit"])]
    if test.empty or test["actual_label"].nunique() < 2:
        return None
    y_true = (test["actual_label"].to_numpy() == "Illicit").astype(np.int64)
    scores = test["illicit_score"].to_numpy(dtype=float)
    predicted = (scores >= threshold).astype(np.int64)
    return {
        "n_evaluated": int(len(test)),
        "illicit_count": int(y_true.sum()),
        "predicted_illicit_count": int(predicted.sum()),
        "precision_illicit": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_illicit": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc_illicit": float(average_precision_score(y_true, scores)),
        "roc_auc_illicit": float(roc_auc_score(y_true, scores)),
        "y_true": y_true,
        "predicted": predicted,
        "scores": scores,
    }


def fmt_pct(value) -> str:
    return "—" if value is None else f"{float(value):.1%}"


metadata = load_metrics()
with st.sidebar:
    st.markdown("## ◉ Forensic Analytics")
    st.caption("ELLIPTIC BITCOIN GRAPH · LORENTZ HYPERBOLIC GNN")
    st.divider()
    uploaded = st.file_uploader(
        "Load real prediction export (.csv)",
        type=["csv"],
        help="Select predictions.csv generated by this repository's export_predictions.py.",
    )
    source_frame = None
    source_error = None
    if uploaded is not None:
        try:
            source_frame = normalize_predictions(pd.read_csv(uploaded))
            st.success("Real model export loaded")
        except Exception as exc:
            source_error = str(exc)
    elif PREDICTIONS_PATH.is_file():
        try:
            source_frame = normalize_predictions(pd.read_csv(PREDICTIONS_PATH))
            st.success("Repository prediction export loaded")
        except Exception as exc:
            source_error = f"Could not load {PREDICTIONS_PATH}: {exc}"
    else:
        st.info("No model-output CSV is bundled yet. See the README to train and export the original Lorentz model.")
    saved_threshold = float(metadata.get("threshold", 0.5))
    threshold = st.slider(
        "Review threshold", min_value=0.0, max_value=1.0, value=float(np.clip(saved_threshold, 0, 1)),
        step=0.0001, format="%.4f",
        help="Controls the displayed flags. The saved threshold is selected on validation labels, not test labels.",
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Graph machine learning · Research prototype</div>
      <h1>Hyperbolic GNN<br>for Forensic Analytics</h1>
      <p>A Lorentz-model graph-attention network for transaction-graph representation and illicit-transaction research using the Elliptic Bitcoin dataset. Explore exported scores and the two-dimensional projection produced by the trained model.</p>
      <span class="pill">Lorentz hyperboloid model</span><span class="pill">Poincaré disk projection</span><span class="pill">Elliptic transaction graph</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if source_error:
    st.error(source_error)

if not metadata and source_frame is None:
    st.warning(
        "This app is configured, but real model outputs have not been exported yet. "
        "Run `notebooks/Train_and_Export_Lorentz_GNN.ipynb` against your Elliptic dataset, "
        "then place the generated predictions and deployment metrics in `artifacts/`. "
        "No placeholder transactions or model scores are generated."
    )

metrics_payload = metadata.get("test_metrics_at_validation_threshold", {}) if metadata else {}
if not metrics_payload and source_frame is not None:
    calculated = calculate_metrics(source_frame, threshold)
    if calculated:
        metrics_payload = calculated

if metadata:
    n_eval = metrics_payload.get("n_evaluated")
    illicit = metrics_payload.get("illicit_count")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Held-out transactions", f"{int(n_eval):,}" if n_eval is not None else "—")
    p2.metric("Illicit prevalence", fmt_pct((int(illicit) / int(n_eval)) if n_eval and illicit is not None else None))
    p3.metric("PR-AUC", f"{float(metrics_payload['pr_auc_illicit']):.4f}" if metrics_payload.get("pr_auc_illicit") is not None else "—")
    p4.metric("ROC-AUC", f"{float(metrics_payload['roc_auc_illicit']):.4f}" if metrics_payload.get("roc_auc_illicit") is not None else "—")
elif source_frame is not None:
    k1, k2, k3 = st.columns(3)
    k1.metric("Transactions in export", f"{len(source_frame):,}")
    k2.metric("Flagged at threshold", f"{int((source_frame['illicit_score'] >= threshold).sum()):,}")
    k3.metric("Unknown labels", f"{int((source_frame['actual_label'] == 'Unknown').sum()):,}")

overview_tab, explorer_tab, evaluation_tab, methodology_tab = st.tabs(["Overview", "Transaction explorer", "Evaluation", "Methodology"])

with overview_tab:
    if source_frame is not None:
        left, right = st.columns(2)
        with left:
            plot_data = source_frame.copy()
            plot_data["predicted_label"] = np.where(plot_data["illicit_score"] >= threshold, "Illicit", "Licit")
            fig = px.histogram(plot_data, x="illicit_score", color="predicted_label", nbins=48, barmode="overlay", opacity=0.78, template="plotly_dark", title="Exported model score distribution")
            fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold {threshold:.4f}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,35,.55)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            by_time = source_frame.groupby("timestep", as_index=False).agg(transactions=("tx_id", "size"), mean_score=("illicit_score", "mean"), flagged=("illicit_score", lambda s: int((s >= threshold).sum())))
            fig = px.line(by_time, x="timestep", y="mean_score", markers=True, template="plotly_dark", title="Mean illicit score by timestep")
            fig.add_hline(y=threshold, line_dash="dash", annotation_text="review threshold")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,35,.55)", yaxis_title="Mean illicit score")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Scores are ranking signals, not calibrated probabilities. A flag indicates a review priority, not proof of criminal activity.")
    else:
        st.subheader("Research overview")
        st.write("The deployed interface is ready to visualize the outputs from the original Lorentz-model pipeline once the real Elliptic graph has been processed.")
        history = read_json(HISTORY_PATH)
        if isinstance(history, list) and history:
            hist = pd.DataFrame(history)
            if "epoch" in hist and "train_loss" in hist:
                fig = go.Figure(go.Scatter(x=hist["epoch"], y=hist["train_loss"], mode="lines+markers", name="Training loss"))
                fig.update_layout(template="plotly_dark", title="Training loss by epoch", xaxis_title="Epoch", yaxis_title="Cross-entropy loss", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,35,.55)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training curves appear here after `training_history.json` is copied into the artifacts folder.")

with explorer_tab:
    if source_frame is None:
        st.info("The transaction explorer requires `artifacts/predictions.csv`, created from the trained Lorentz model. It is intentionally empty rather than showing synthetic examples.")
    else:
        st.subheader("Transaction search and triage")
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            query = st.text_input("Transaction ID contains", placeholder="Search transaction ID")
        with f2:
            tmin, tmax = int(source_frame.timestep.min()), int(source_frame.timestep.max())
            timesteps = st.slider("Timestep range", min_value=tmin, max_value=tmax, value=(tmin, tmax))
        with f3:
            risk_options = ["All", "Top 1% score", "Top 10% score", "Remaining"]
            risk = st.selectbox("Score band", risk_options)
        filtered = source_frame[source_frame["timestep"].between(timesteps[0], timesteps[1]) & source_frame["tx_id"].str.contains(query, case=False, regex=False)]
        if risk != "All" and "risk_band" in filtered.columns:
            filtered = filtered[filtered["risk_band"] == risk]
        filtered = filtered.assign(predicted_label=np.where(filtered["illicit_score"] >= threshold, "Illicit", "Licit"))
        display_columns = [c for c in ["tx_id", "timestep", "actual_label", "illicit_score", "predicted_label", "risk_band", "degree"] if c in filtered.columns]
        st.caption(f"Showing {len(filtered):,} of {len(source_frame):,} rows")
        st.dataframe(filtered.sort_values("illicit_score", ascending=False)[display_columns].head(1000), use_container_width=True, hide_index=True)
        st.download_button("Download filtered rows", data=filtered[display_columns].to_csv(index=False).encode("utf-8"), file_name="filtered_transaction_scores.csv", mime="text/csv")
        if {"embedding_x", "embedding_y"}.issubset(filtered.columns):
            emb = filtered.dropna(subset=["embedding_x", "embedding_y"])
            if not emb.empty:
                fig = px.scatter(emb.head(5000), x="embedding_x", y="embedding_y", color="actual_label", hover_data=["tx_id", "timestep", "illicit_score"], template="plotly_dark", title="Model embedding projected into the Poincaré disk")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,35,.55)", xaxis_scaleanchor="y", xaxis_scaleratio=1)
                fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line_color="rgba(200,220,255,.65)")
                st.plotly_chart(fig, use_container_width=True)

with evaluation_tab:
    st.subheader("Held-out labelled evaluation · timesteps 42–49")
    active = calculate_metrics(source_frame, threshold) if source_frame is not None else None
    shown = active or metrics_payload
    if shown:
        st.caption(f"{'Metrics recomputed from the loaded real export' if active is not None else 'Saved metrics at the validation-selected threshold'}.")
        cols = st.columns(5)
        cols[0].metric("Precision · illicit", f"{float(shown.get('precision_illicit', 0)):.4f}")
        cols[1].metric("Recall · illicit", f"{float(shown.get('recall_illicit', 0)):.4f}")
        cols[2].metric("F1 · illicit", f"{float(shown.get('f1_illicit', 0)):.4f}")
        cols[3].metric("PR-AUC", f"{float(shown.get('pr_auc_illicit', 0)):.4f}")
        cols[4].metric("ROC-AUC", f"{float(shown.get('roc_auc_illicit', 0)):.4f}" if shown.get("roc_auc_illicit") is not None else "—")
        if active is not None:
            cm = confusion_matrix(active["y_true"], active["predicted"], labels=[1, 0])
            fig = px.imshow(cm, text_auto=True, x=["Predicted illicit", "Predicted licit"], y=["Actual illicit", "Actual licit"], color_continuous_scale="Teal", template="plotly_dark", title="Confusion matrix")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,35,.55)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trained model evaluation has been exported yet. Run the Colab notebook to create genuine validation/test metrics.")
    st.warning("This is a research prototype. A flag is not proof of illicit activity, and any reported metrics must be interpreted in light of the transductive graph split and class imbalance.")

with methodology_tab:
    st.subheader("Model and experimental design")
    st.markdown(
        """
        **Geometry.** The original project uses the **Lorentz (hyperboloid) model** for message passing and maps the compact representation to two dimensions for a Poincaré-disk visualization.

        **Graph model.** Two graph-attention layers perform neighbourhood aggregation in the tangent space. A hyperbolic linear layer creates the compact representation, followed by the illicit/licit classifier.

        **Dataset.** The Elliptic Bitcoin transaction graph supplies 165 numeric transaction features. Original class `1` is mapped to illicit, class `2` to licit, and unknown labels are excluded from supervised training and evaluation. Timestep is used only for chronological masks, not as a model feature.

        **Evaluation protocol.** Labelled timesteps 1–34 train the model, 35–41 select a decision threshold, and 42–49 form the held-out labelled test window. All available graph edges participate in message passing, so this is a transductive chronological-label split, not strict future-isolated generalization.

        **Limitations.** Thresholds must be chosen on validation data. Scores are not calibrated risk probabilities, and further baseline comparisons, repeated-seed experiments, and stricter temporal or graph-isolated evaluation are required before strong research claims.
        """
    )

st.divider()
st.markdown('<p class="subtle">RESEARCH PROTOTYPE · PYTORCH GEOMETRIC · GEOOPT · LORENTZ GEOMETRY · STREAMLIT</p>', unsafe_allow_html=True)
