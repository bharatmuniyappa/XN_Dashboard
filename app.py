import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="XN Capstone Dashboard",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path("module11_output")
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
METRICS_JSON = BASE_DIR / "module11_metrics_summary.json"
METRICS_TXT = BASE_DIR / "module11_metrics_summary.txt"

# ------------------------------------------------------------
# HELPERS
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

def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.error(f"Missing image: {path.name}")

def show_table(path: Path, title: str):
    df = load_csv(path)
    if df is not None:
        st.markdown(f"**{title}**")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.error(f"Missing table: {path.name}")

# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------
metrics = load_json(METRICS_JSON)

# tables actually present
threshold_df = load_csv(TAB_DIR / "Table1_Threshold_Sweep_Enhanced_Logistic.csv")
calibration_df = load_csv(TAB_DIR / "Table2_Calibration_Summary_BoostedTree.csv")
comparison_df = load_csv(TAB_DIR / "Table3_Model_Comparison.csv")

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("XN Dashboard")
st.sidebar.markdown("**Emergency Department Utilization Risk**")
page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Baseline Logistic Model",
        "Enhanced Logistic Model",
        "Boosted Tree Benchmark",
        "Model Comparison",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Suggested report screenshots:**")
st.sidebar.markdown("1. Overview")
st.sidebar.markdown("2. Enhanced Logistic Model")
st.sidebar.markdown("3. Boosted Tree Benchmark")
st.sidebar.markdown("4. Model Comparison")

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("Emergency Department Utilization Risk Dashboard")
st.caption(
    "Pooled MEPS adult non-cancer cohort | Baseline logistic vs enhanced logistic vs boosted tree"
)

# ------------------------------------------------------------
# OVERVIEW PAGE
# ------------------------------------------------------------
if page == "Overview":
    st.subheader("Project Overview")

    st.write(
        """
        This dashboard summarizes the final phase of the capstone project, which predicts
        follow-up-year emergency department utilization among U.S. adults without cancer
        using pooled MEPS longitudinal panels.
        """
    )

    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Outcome", metrics.get("outcome", "ER_ANY_Y2"))
        c2.metric("Outcome Prevalence", f"{metrics.get('outcome_prevalence', 0):.3f}")
        c3.metric("Modeling Sample", f"{metrics.get('n_modeling', 0):,}")
        best_auc = max(
            metrics["baseline_logistic"]["roc_auc"],
            metrics["enhanced_logistic"]["roc_auc"],
            metrics["boosted_tree"]["roc_auc"],
        )
        c4.metric("Best ROC-AUC", f"{best_auc:.4f}")

        st.markdown("### Model Performance Snapshot")
        a, b, c = st.columns(3)

        with a:
            st.markdown("#### Baseline Logistic")
            st.write(f"ROC-AUC: **{metrics['baseline_logistic']['roc_auc']:.4f}**")
            st.write(f"PR-AUC: **{metrics['baseline_logistic']['pr_auc']:.4f}**")

        with b:
            st.markdown("#### Enhanced Logistic")
            st.write(f"ROC-AUC: **{metrics['enhanced_logistic']['roc_auc']:.4f}**")
            st.write(f"PR-AUC: **{metrics['enhanced_logistic']['pr_auc']:.4f}**")

        with c:
            st.markdown("#### Boosted Tree")
            st.write(f"ROC-AUC: **{metrics['boosted_tree']['roc_auc']:.4f}**")
            st.write(f"PR-AUC: **{metrics['boosted_tree']['pr_auc']:.4f}**")
            st.write(f"Brier Score: **{metrics['boosted_tree']['brier']:.4f}**")

        st.info(
            "The enhanced logistic model and boosted tree benchmark performed very similarly. "
            "This supports using the enhanced logistic model as the preferred interpretable model."
        )

    if METRICS_TXT.exists():
        st.markdown("### Metrics Summary")
        st.code(METRICS_TXT.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# BASELINE PAGE
# ------------------------------------------------------------
elif page == "Baseline Logistic Model":
    st.subheader("Baseline Logistic Regression")

    if metrics is not None:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['baseline_logistic']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['baseline_logistic']['pr_auc']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        show_image(
            FIG_DIR / "Figure1_ROC_Baseline_Logistic.png",
            "ROC Curve - Baseline Logistic Model",
        )
    with col2:
        show_image(
            FIG_DIR / "Figure2_Confusion_Baseline_Logistic.png",
            "Confusion Matrix - Baseline Logistic Model",
        )

    st.write(
        """
        The baseline logistic regression model uses demographic and access-related predictors only.
        It serves as the first interpretable benchmark for identifying follow-up-year ER utilization risk.
        """
    )

# ------------------------------------------------------------
# ENHANCED PAGE
# ------------------------------------------------------------
elif page == "Enhanced Logistic Model":
    st.subheader("Enhanced Logistic Regression")

    if metrics is not None:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['enhanced_logistic']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['enhanced_logistic']['pr_auc']:.4f}")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        show_image(
            FIG_DIR / "Figure3_ROC_Enhanced_Logistic.png",
            "ROC Curve - Enhanced Logistic Model",
        )
    with row1_col2:
        show_image(
            FIG_DIR / "Figure4_PR_Enhanced_Logistic.png",
            "Precision-Recall Curve - Enhanced Logistic Model",
        )

    st.markdown("---")

    show_image(
        FIG_DIR / "Figure5_Threshold_Sweep_Enhanced_Logistic.png",
        "Threshold Sweep - Enhanced Logistic Model",
    )

    show_table(
        TAB_DIR / "Table1_Threshold_Sweep_Enhanced_Logistic.csv",
        "Threshold Sweep Results",
    )

    st.write(
        """
        This model extends the baseline logistic model by adding prior-year expenditure features.
        It improves discrimination and provides stronger practical value for risk stratification.
        """
    )

# ------------------------------------------------------------
# BOOSTED TREE PAGE
# ------------------------------------------------------------
elif page == "Boosted Tree Benchmark":
    st.subheader("Boosted Tree Benchmark")

    if metrics is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{metrics['boosted_tree']['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['boosted_tree']['pr_auc']:.4f}")
        c3.metric("Brier Score", f"{metrics['boosted_tree']['brier']:.4f}")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        show_image(
            FIG_DIR / "Figure6_ROC_BoostedTree.png",
            "ROC Curve - Boosted Tree Benchmark",
        )
    with row1_col2:
        show_image(
            FIG_DIR / "Figure7_PR_BoostedTree.png",
            "Precision-Recall Curve - Boosted Tree Benchmark",
        )

    st.markdown("---")

    show_image(
        FIG_DIR / "Figure8_Calibration_BoostedTree.png",
        "Calibration Plot - Boosted Tree Benchmark",
    )

    show_table(
        TAB_DIR / "Table2_Calibration_Summary_BoostedTree.csv",
        "Calibration Summary",
    )

    st.write(
        """
        The boosted tree benchmark was used to test whether a more flexible nonlinear model
        could outperform the enhanced logistic model. The results show only marginal improvement.
        """
    )

# ------------------------------------------------------------
# COMPARISON PAGE
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.subheader("Model Comparison")

    show_image(
        FIG_DIR / "Figure9_Model_Comparison.png",
        "Model Performance Comparison",
    )

    show_table(
        TAB_DIR / "Table3_Model_Comparison.csv",
        "Model Comparison Table",
    )

    if metrics is not None:
        baseline_auc = metrics["baseline_logistic"]["roc_auc"]
        enhanced_auc = metrics["enhanced_logistic"]["roc_auc"]
        boosted_auc = metrics["boosted_tree"]["roc_auc"]

        st.success(
            f"The enhanced logistic model improves ROC-AUC by {enhanced_auc - baseline_auc:.4f} "
            f"over the baseline model, while the boosted tree improves by only "
            f"{boosted_auc - enhanced_auc:.4f} over the enhanced logistic model."
        )

    st.write(
        """
        The comparison shows that the enhanced logistic model and boosted tree benchmark perform
        almost the same. Because of this, the enhanced logistic model is the more practical option
        due to its interpretability and nearly equivalent performance.
        """
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit for XN Capstone Dashboard Presentation.")
