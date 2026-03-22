import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="XN Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path("module11_output")
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
METRICS_JSON = BASE_DIR / "module11_metrics_summary.json"
METRICS_TXT = BASE_DIR / "module11_metrics_summary.txt"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None

def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path.name}")

def show_table_if_exists(path: Path, title: str):
    df = load_csv(path)
    if df is not None:
        st.markdown(f"**{title}**")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"Table not found: {path.name}")

# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
metrics = load_json(METRICS_JSON)

threshold_df = load_csv(TAB_DIR / "Table1_Threshold_Sweep_Enhanced_Logistic.csv")
calibration_df = load_csv(TAB_DIR / "Table2_Calibration_Summary_BoostedTree.csv")
comparison_df = load_csv(TAB_DIR / "Table3_Model_Comparison.csv")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("XN Dashboard")
st.sidebar.markdown("MEPS pooled-panel ER utilization modeling dashboard")

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Baseline Logistic Model",
        "Enhanced Logistic Model",
        "Boosted Tree Benchmark",
        "Model Comparison",
        "Files Available"
    ]
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Emergency Department Utilization Risk Dashboard")
st.caption(
    "Adult non-cancer pooled MEPS cohort with baseline logistic, enhanced logistic, and boosted tree benchmark results."
)

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if section == "Overview":
    st.subheader("Project Overview")

    st.write(
        "This dashboard presents the final modeling results for predicting follow-up-year emergency department utilization "
        "among adults without cancer using pooled MEPS longitudinal panels. The analysis compares a baseline logistic regression model, "
        "an enhanced logistic regression model with prior-year spending, and a boosted tree benchmark."
    )

    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Outcome", metrics.get("outcome", "ER_ANY_Y2"))
        c2.metric("Outcome prevalence", f"{metrics.get('outcome_prevalence', 0):.3f}")
        c3.metric("Modeling sample", f"{metrics.get('n_modeling', 0):,}")
        c4.metric("Best ROC-AUC", f"{max(metrics['baseline_logistic']['roc_auc'], metrics['enhanced_logistic']['roc_auc'], metrics['boosted_tree']['roc_auc']):.4f}")

        st.markdown("### Performance Summary")
        c5, c6, c7 = st.columns(3)

        with c5:
            st.markdown("**Baseline Logistic**")
            st.write(f"ROC-AUC: {metrics['baseline_logistic']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['baseline_logistic']['pr_auc']:.4f}")

        with c6:
            st.markdown("**Enhanced Logistic**")
            st.write(f"ROC-AUC: {metrics['enhanced_logistic']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['enhanced_logistic']['pr_auc']:.4f}")

        with c7:
            st.markdown("**Boosted Tree**")
            st.write(f"ROC-AUC: {metrics['boosted_tree']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['boosted_tree']['pr_auc']:.4f}")
            st.write(f"Brier Score: {metrics['boosted_tree']['brier']:.4f}")

        st.info(
            "The enhanced logistic model and boosted tree benchmark performed very similarly, "
            "suggesting that the logistic model provides a strong balance between interpretability and predictive power."
        )
    else:
        st.warning("Metrics summary JSON not found.")

# ------------------------------------------------------------
# Baseline Logistic Model
# ------------------------------------------------------------
elif section == "Baseline Logistic Model":
    st.subheader("Baseline Logistic Model")

    if metrics is not None:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['baseline_logistic']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['baseline_logistic']['pr_auc']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        show_image_if_exists(
            FIG_DIR / "Figure1_ROC_Baseline_Logistic.png",
            "Figure 1. ROC Curve - Baseline Logistic Model"
        )

    with col2:
        show_image_if_exists(
            FIG_DIR / "Figure2_Confusion_Baseline_Logistic.png",
            "Figure 2. Confusion Matrix - Baseline Logistic Model"
        )

    st.write(
        "This baseline model uses demographic and access-related predictors only. "
        "It provides an interpretable benchmark and establishes the starting point for model comparison."
    )

# ------------------------------------------------------------
# Enhanced Logistic Model
# ------------------------------------------------------------
elif section == "Enhanced Logistic Model":
    st.subheader("Enhanced Logistic Model")

    if metrics is not None:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['enhanced_logistic']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['enhanced_logistic']['pr_auc']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        show_image_if_exists(
            FIG_DIR / "Figure3_ROC_Enhanced_Logistic.png",
            "Figure 3. ROC Curve - Enhanced Logistic Model"
        )

    with col2:
        show_image_if_exists(
            FIG_DIR / "Figure4_PR_Enhanced_Logistic.png",
            "Figure 4. Precision-Recall Curve - Enhanced Logistic Model"
        )

    show_image_if_exists(
        FIG_DIR / "Figure5_Threshold_Sweep_Enhanced_Logistic.png",
        "Figure 5. Threshold Sweep - Enhanced Logistic Model"
    )

    show_table_if_exists(
        TAB_DIR / "Table1_Threshold_Sweep_Enhanced_Logistic.csv",
        "Table 1. Threshold Sweep Results"
    )

    st.write(
        "This model extends the baseline logistic model by incorporating prior-year spending features. "
        "It improves overall discrimination and provides a more practical risk stratification framework."
    )

# ------------------------------------------------------------
# Boosted Tree Benchmark
# ------------------------------------------------------------
elif section == "Boosted Tree Benchmark":
    st.subheader("Boosted Tree Benchmark")

    if metrics is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{metrics['boosted_tree']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['boosted_tree']['pr_auc']:.4f}")
        c3.metric("Brier Score", f"{metrics['boosted_tree']['brier']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        show_image_if_exists(
            FIG_DIR / "Figure6_ROC_BoostedTree.png",
            "Figure 6. ROC Curve - Boosted Tree Benchmark"
        )

    with col2:
        show_image_if_exists(
            FIG_DIR / "Figure7_PR_BoostedTree.png",
            "Figure 7. Precision-Recall Curve - Boosted Tree Benchmark"
        )

    show_image_if_exists(
        FIG_DIR / "Figure8_Calibration_BoostedTree.png",
        "Figure 8. Calibration Plot - Boosted Tree Benchmark"
    )

    show_table_if_exists(
        TAB_DIR / "Table2_Calibration_Summary_BoostedTree.csv",
        "Table 2. Calibration Summary"
    )

    st.write(
        "The boosted tree model serves as a nonlinear benchmark. "
        "It produced only marginal improvement over the enhanced logistic model, "
        "which supports retaining the logistic model as the main interpretable approach."
    )

# ------------------------------------------------------------
# Model Comparison
# ------------------------------------------------------------
elif section == "Model Comparison":
    st.subheader("Model Comparison")

    show_image_if_exists(
        FIG_DIR / "Figure9_Model_Comparison.png",
        "Figure 9. Model Performance Comparison"
    )

    show_table_if_exists(
        TAB_DIR / "Table3_Model_Comparison.csv",
        "Table 3. Model Comparison"
    )

    if metrics is not None:
        baseline_auc = metrics["baseline_logistic"]["roc_auc"]
        enhanced_auc = metrics["enhanced_logistic"]["roc_auc"]
        boosted_auc = metrics["boosted_tree"]["roc_auc"]

        st.info(
            f"Enhanced logistic improves ROC-AUC by {enhanced_auc - baseline_auc:.4f} over baseline. "
            f"Boosted tree improves ROC-AUC by only {boosted_auc - enhanced_auc:.4f} over enhanced logistic."
        )

    st.write(
        "The comparison highlights that the enhanced logistic model and boosted tree benchmark perform almost the same. "
        "This makes the enhanced logistic model especially attractive because it is easier to explain and interpret."
    )

# ------------------------------------------------------------
# Files Available
# ------------------------------------------------------------
elif section == "Summary":
    st.subheader("XN Project")

    if METRICS_TXT.exists():
        st.markdown("**Metrics Summary**")
        st.code(METRICS_TXT.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit for the XN capstone dashboard.")