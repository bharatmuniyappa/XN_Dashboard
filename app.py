# app.py
# ============================================================
# XN Capstone Dashboard - Streamlit App
# ------------------------------------------------------------
# Purpose:
# Display the final results from the MEPS pooled-panel project:
# - Cohort summary
# - Descriptive EDA
# - Baseline logistic model
# - Enhanced logistic model
# - Boosted tree benchmark
# - Threshold and calibration outputs
#
# Expected folder structure:
# module11_output/
#   ├── figures/
#   ├── tables/
#   ├── module11_metrics_summary.json
#   └── module11_metrics_summary.txt
# ============================================================

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="XN Capstone Dashboard",
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
# Helper functions
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
        st.info(f"Image not found: {path.name}")

def show_table_if_exists(path: Path, title: str):
    df = load_csv(path)
    if df is not None:
        st.markdown(f"**{title}**")
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"Table not found: {path.name}")

def metric_card(label: str, value, delta_text=None):
    col = st.container()
    if delta_text is None:
        col.metric(label, value)
    else:
        col.metric(label, value, delta=delta_text)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
metrics = load_json(METRICS_JSON)

table1_panel_summary = load_csv(TAB_DIR / "Table1_Panel_Summary.csv")
table2_missingness = load_csv(TAB_DIR / "Table2_Missingness_ModelDF.csv")
table3_ageband = load_csv(TAB_DIR / "Table3_Outcome_By_AgeBand.csv")
table4_threshold = load_csv(TAB_DIR / "Table4_Threshold_Sweep.csv")
table5_calibration = load_csv(TAB_DIR / "Table5_Calibration_Summary.csv")
table6_model_comp = load_csv(TAB_DIR / "Table6_Model_Comparison.csv")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("XN Dashboard")
st.sidebar.markdown("Final project dashboard for pooled MEPS longitudinal ER-risk modeling.")

section = st.sidebar.radio(
    "Go to section",
    [
        "Overview",
        "Cohort & Data Quality",
        "Exploratory Analysis",
        "Baseline Model",
        "Enhanced Model",
        "Boosted Tree",
        "Model Comparison",
        "Report Assets"
    ]
)

show_tables = st.sidebar.checkbox("Show data tables", value=True)
show_images = st.sidebar.checkbox("Show figures", value=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Emergency Department Utilization Risk Dashboard")
st.caption(
    "Pooled adult non-cancer MEPS cohort with baseline logistic, enhanced logistic, "
    "and boosted tree benchmark results."
)

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if section == "Overview":
    st.subheader("Project Summary")

    st.write(
        "This dashboard summarizes the final modeling stage of the project, where pooled longitudinal "
        "MEPS panels were used to predict follow-up-year emergency department utilization among adults "
        "without cancer. The modeling workflow compared a baseline logistic regression benchmark, an "
        "enhanced logistic model with prior-year spending, and a boosted tree benchmark."
    )

    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Outcome", metrics.get("outcome", "ER_ANY_Y2"))
        with c2:
            metric_card("Outcome prevalence", f"{metrics.get('outcome_prevalence', 0):.3f}")
        with c3:
            metric_card("Modeling sample", f"{metrics.get('n_modeling', 0):,}")
        with c4:
            metric_card(
                "Best ROC-AUC",
                f"{max(metrics['baseline_logistic']['roc_auc'], metrics['enhanced_logistic']['roc_auc'], metrics['boosted_tree']['roc_auc']):.4f}"
            )

        st.markdown("### Model Performance Snapshot")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("**Baseline Logistic**")
            st.write(f"ROC-AUC: {metrics['baseline_logistic']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['baseline_logistic']['pr_auc']:.4f}")

        with m2:
            st.markdown("**Enhanced Logistic**")
            st.write(f"ROC-AUC: {metrics['enhanced_logistic']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['enhanced_logistic']['pr_auc']:.4f}")

        with m3:
            st.markdown("**Boosted Tree**")
            st.write(f"ROC-AUC: {metrics['boosted_tree']['roc_auc']:.4f}")
            st.write(f"PR-AUC: {metrics['boosted_tree']['pr_auc']:.4f}")
            st.write(f"Brier: {metrics['boosted_tree']['brier']:.4f}")

        st.success(
            "Interpretation: the enhanced logistic model and boosted tree performed very similarly, "
            "which supports keeping the logistic model as the main interpretable model."
        )
    else:
        st.warning("Metrics JSON was not found. Run the notebook first to generate outputs.")

# ------------------------------------------------------------
# Cohort & Data Quality
# ------------------------------------------------------------
elif section == "Cohort & Data Quality":
    st.subheader("Cohort Construction and Data Quality")

    st.write(
        "This section summarizes the pooled cohort after adult restriction and cancer exclusion, "
        "along with missingness checks for the final modeling dataframe."
    )

    if show_tables:
        show_table_if_exists(TAB_DIR / "Table1_Panel_Summary.csv", "Table 1. Panel Summary")
        show_table_if_exists(TAB_DIR / "Table2_Missingness_ModelDF.csv", "Table 2. Missingness in Final Modeling Data")

# ------------------------------------------------------------
# Exploratory Analysis
# ------------------------------------------------------------
elif section == "Exploratory Analysis":
    st.subheader("Exploratory Analysis")

    st.write(
        "These plots summarize baseline expenditure patterns and subgroup differences in follow-up-year "
        "ER utilization."
    )

    col1, col2 = st.columns(2)

    with col1:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure1_PriorYear_Expenditure_Distribution.png",
                "Figure 1. Prior-Year Expenditure Distribution"
            )

    with col2:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure2_Outcome_By_AgeBand.png",
                "Figure 2. Outcome Rate by Age Band"
            )

    if show_tables:
        show_table_if_exists(
            TAB_DIR / "Table3_Outcome_By_AgeBand.csv",
            "Table 3. Outcome Rate by Age Band"
        )

# ------------------------------------------------------------
# Baseline Model
# ------------------------------------------------------------
elif section == "Baseline Model":
    st.subheader("Baseline Logistic Model")

    st.write(
        "The baseline logistic regression model used demographic and access-related baseline predictors "
        "as the initial interpretable benchmark."
    )

    if metrics is not None:
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Baseline ROC-AUC", f"{metrics['baseline_logistic']['roc_auc']:.4f}")
        with c2:
            metric_card("Baseline PR-AUC", f"{metrics['baseline_logistic']['pr_auc']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure3_ROC_Baseline_Logistic.png",
                "Figure 3. ROC Curve - Baseline Logistic Model"
            )

    with col2:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure4_Confusion_Baseline_Logistic.png",
                "Figure 4. Confusion Matrix - Baseline Logistic Model"
            )

# ------------------------------------------------------------
# Enhanced Model
# ------------------------------------------------------------
elif section == "Enhanced Model":
    st.subheader("Enhanced Logistic Model")

    st.write(
        "The enhanced logistic model extends the baseline model by adding prior-year expenditure "
        "features, which improved discrimination and recall."
    )

    if metrics is not None:
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Enhanced ROC-AUC", f"{metrics['enhanced_logistic']['roc_auc']:.4f}")
        with c2:
            metric_card("Enhanced PR-AUC", f"{metrics['enhanced_logistic']['pr_auc']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure5_ROC_Enhanced_Logistic.png",
                "Figure 5. ROC Curve - Enhanced Logistic Model"
            )

    with col2:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure6_PR_Enhanced_Logistic.png",
                "Figure 6. Precision-Recall Curve - Enhanced Logistic Model"
            )

    if show_images:
        show_image_if_exists(
            FIG_DIR / "Figure7_Threshold_Sweep.png",
            "Figure 7. Threshold Sweep - Enhanced Logistic Model"
        )

    if show_tables:
        show_table_if_exists(
            TAB_DIR / "Table4_Threshold_Sweep.csv",
            "Table 4. Threshold Sweep Results"
        )

# ------------------------------------------------------------
# Boosted Tree
# ------------------------------------------------------------
elif section == "Boosted Tree":
    st.subheader("Boosted Tree Benchmark")

    st.write(
        "The boosted tree benchmark was used to test whether a more flexible nonlinear model could "
        "meaningfully outperform the enhanced logistic model."
    )

    if metrics is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Boosted ROC-AUC", f"{metrics['boosted_tree']['roc_auc']:.4f}")
        with c2:
            metric_card("Boosted PR-AUC", f"{metrics['boosted_tree']['pr_auc']:.4f}")
        with c3:
            metric_card("Boosted Brier", f"{metrics['boosted_tree']['brier']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure8_ROC_BoostedTree.png",
                "Figure 8. ROC Curve - Boosted Tree Benchmark"
            )

    with col2:
        if show_images:
            show_image_if_exists(
                FIG_DIR / "Figure9_PR_BoostedTree.png",
                "Figure 9. Precision-Recall Curve - Boosted Tree Benchmark"
            )

    if show_images:
        show_image_if_exists(
            FIG_DIR / "Figure10_Calibration_Plot.png",
            "Figure 10. Calibration Plot - Boosted Tree Benchmark"
        )

    if show_tables:
        show_table_if_exists(
            TAB_DIR / "Table5_Calibration_Summary.csv",
            "Table 5. Calibration Summary"
        )

# ------------------------------------------------------------
# Model Comparison
# ------------------------------------------------------------
elif section == "Model Comparison":
    st.subheader("Model Comparison")

    st.write(
        "This section compares the baseline logistic, enhanced logistic, and boosted tree models. "
        "The main practical takeaway is that the enhanced logistic model performs nearly as well as "
        "the boosted tree while remaining easier to interpret."
    )

    if show_tables:
        show_table_if_exists(
            TAB_DIR / "Table6_Model_Comparison.csv",
            "Table 6. Model Comparison"
        )

    if show_images:
        show_image_if_exists(
            FIG_DIR / "Figure11_Model_Comparison.png",
            "Figure 11. Model Performance Comparison"
        )

    if metrics is not None:
        baseline_auc = metrics["baseline_logistic"]["roc_auc"]
        enhanced_auc = metrics["enhanced_logistic"]["roc_auc"]
        boosted_auc = metrics["boosted_tree"]["roc_auc"]

        st.info(
            f"Enhanced logistic improves ROC-AUC by {enhanced_auc - baseline_auc:.4f} over baseline, "
            f"while boosted tree improves by only {boosted_auc - enhanced_auc:.4f} over enhanced logistic."
        )

# ------------------------------------------------------------
# Report Assets
# ------------------------------------------------------------
elif section == "Report Assets":
    st.subheader("Saved Report Assets")

    st.write(
        "This section lists the available figures and tables that can be inserted into the final report."
    )

    figure_files = sorted([p.name for p in FIG_DIR.glob("*.png")]) if FIG_DIR.exists() else []
    table_files = sorted([p.name for p in TAB_DIR.glob("*.csv")]) if TAB_DIR.exists() else []

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Figures**")
        if figure_files:
            for f in figure_files:
                st.write("-", f)
        else:
            st.write("No figure files found.")

    with c2:
        st.markdown("**Tables**")
        if table_files:
            for f in table_files:
                st.write("-", f)
        else:
            st.write("No table files found.")

    if METRICS_TXT.exists():
        st.markdown("**Metrics Summary Text**")
        st.code(METRICS_TXT.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built in Streamlit for the XN capstone project dashboard.")