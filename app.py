"""
AeroLens - Airline Customer Experience Intelligence Platform
=============================================================
Streamlit application entry point.

Tagline: Turning customer conversations into actionable insights.

This file implements the complete enterprise-grade UI with four sections:
  1. Executive Overview   - KPIs, trends, sentiment distribution
  2. Complaint Intelligence - Category analysis, route hotspots, comparisons
  3. Feedback Explorer    - Search, filter, inspect individual feedback
  4. Feedback Analyser    - Single review + batch CSV prediction
"""

import os
import sys
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add src/ to path so we can import our modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import load_models, predict_single, predict_batch, assign_priority
from preprocessing import clean_text

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroLens | Airline CX Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for enterprise SaaS appearance
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #132240 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #e2e8f0 !important;
    }

    /* KPI Card Styling */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-navy { color: #0f2b5b; }
    .kpi-teal { color: #0d9488; }
    .kpi-red  { color: #dc2626; }
    .kpi-amber { color: #d97706; }

    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f2b5b;
        border-bottom: 3px solid #0d9488;
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 10px;
    }

    /* Brand header */
    .brand-header {
        text-align: center;
        padding: 12px 0;
    }
    .brand-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: 0.05em;
    }
    .brand-tagline {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 2px;
    }
    .brand-icon {
        font-size: 2rem;
        margin-bottom: 4px;
    }

    /* Data table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 500;
    }

    /* Metric card overrides */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
    }

    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #0d9488;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #334155;
    }

    /* Priority badges */
    .priority-high { color: #dc2626; font-weight: 600; }
    .priority-medium { color: #d97706; font-weight: 600; }
    .priority-low { color: #16a34a; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, color_class: str = "kpi-navy") -> str:
    """Return HTML for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
    </div>
    """


def section_header(text: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the cleaned feedback dataset."""
    path = os.path.join(PROJECT_ROOT, "data", "processed", "feedback_cleaned.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["feedback_date"])
    return df


@st.cache_resource
def get_models():
    """Load trained ML models (cached across reruns)."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    return load_models(models_dir)


def load_eval_results() -> dict:
    """Load evaluation metrics from outputs/evaluation_results.json."""
    path = os.path.join(PROJECT_ROOT, "outputs", "evaluation_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_template_csv() -> str:
    """Generate a CSV template string for data upload."""
    template = pd.DataFrame({
        "feedback_id": ["EXAMPLE001"],
        "feedback_date": ["2025-06-15"],
        "airline": ["IndiGo"],
        "route": ["DEL-BOM"],
        "source": ["Customer Survey"],
        "review_text": ["The flight was delayed by 3 hours with no communication."],
        "satisfaction_rating": [2],
    })
    return template.to_csv(index=False)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">&#9992;</div>
        <div class="brand-name">AeroLens</div>
        <div class="brand-tagline">Turning customer conversations<br>into actionable insights</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Complaint Intelligence",
            "Feedback Explorer",
            "Feedback Analyser",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Load data
    df_all = load_dataset()

    if len(df_all) > 0:
        # Airline filter
        airlines = ["All Airlines"] + sorted(df_all["airline"].unique().tolist())
        selected_airline = st.selectbox("Select Airline", airlines)

        if selected_airline != "All Airlines":
            df = df_all[df_all["airline"] == selected_airline].copy()
        else:
            df = df_all.copy()

        # Date range filter
        min_date = df_all["feedback_date"].min().date()
        max_date = df_all["feedback_date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d_start, d_end = date_range
            df = df[
                (df["feedback_date"].dt.date >= d_start)
                & (df["feedback_date"].dt.date <= d_end)
            ]

        st.markdown("---")
        st.markdown(
            f'<div style="color:#94a3b8; font-size:0.78rem; text-align:center;">'
            f'Showing <b style="color:#e2e8f0;">{len(df):,}</b> of '
            f'<b style="color:#e2e8f0;">{len(df_all):,}</b> records'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame()
        st.warning("No data loaded. Run the ML pipeline first.")

    st.markdown("---")
    st.markdown(
        '<div style="color:#475569; font-size:0.7rem; text-align:center;">'
        'Dataset is 100% synthetic.<br>Not real airline performance data.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Utility: Matplotlib chart styling
# ---------------------------------------------------------------------------

COLORS = {
    "navy": "#0f2b5b",
    "teal": "#0d9488",
    "light_teal": "#5eead4",
    "red": "#dc2626",
    "amber": "#d97706",
    "green": "#16a34a",
    "slate": "#64748b",
    "light_gray": "#f1f5f9",
    "blue": "#2563eb",
    "indigo": "#4f46e5",
    "rose": "#e11d48",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "cyan": "#0891b2",
    "emerald": "#059669",
    "pink": "#db2777",
}

CHART_PALETTE = [
    COLORS["teal"], COLORS["navy"], COLORS["amber"],
    COLORS["red"], COLORS["blue"], COLORS["purple"],
    COLORS["emerald"], COLORS["orange"],
]

SENTIMENT_COLORS = {
    "Positive": COLORS["teal"],
    "Negative": COLORS["red"],
    "Neutral": COLORS["slate"],
}

PRIORITY_COLORS = {
    "High": COLORS["red"],
    "Medium": COLORS["amber"],
    "Low": COLORS["teal"],
}

def style_chart(ax, title="", xlabel="", ylabel=""):
    """Apply consistent chart styling."""
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["navy"], pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=COLORS["slate"])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS["slate"])
    ax.tick_params(colors=COLORS["slate"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")


# ===========================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ===========================================================================

if page == "Executive Overview":
    st.markdown(
        '<h1 style="color:#0f2b5b; font-weight:700; margin-bottom:4px;">'
        '&#9992; Executive Overview</h1>'
        '<p style="color:#64748b; margin-top:0;">High-level customer experience metrics and trends</p>',
        unsafe_allow_html=True,
    )

    if len(df) == 0:
        st.info("No data available. Please upload feedback or run the data pipeline.")
    else:
        # --- KPI Row ---
        total = len(df)
        neg_count = len(df[df["sentiment"] == "Negative"])
        pos_count = len(df[df["sentiment"] == "Positive"])
        neu_count = len(df[df["sentiment"] == "Neutral"])
        neg_rate = neg_count / total * 100 if total > 0 else 0
        high_priority = len(df[df["priority"] == "High"])
        avg_rating = df["satisfaction_rating"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(kpi_card("Total Feedback", f"{total:,}", "kpi-navy"), unsafe_allow_html=True)
        with col2:
            st.markdown(kpi_card("Negative Rate", f"{neg_rate:.1f}%", "kpi-red"), unsafe_allow_html=True)
        with col3:
            st.markdown(kpi_card("High Priority", f"{high_priority:,}", "kpi-amber"), unsafe_allow_html=True)
        with col4:
            st.markdown(kpi_card("Avg. Satisfaction", f"{avg_rating:.1f}/5", "kpi-teal"), unsafe_allow_html=True)
        with col5:
            pos_rate = pos_count / total * 100 if total > 0 else 0
            st.markdown(kpi_card("Positive Rate", f"{pos_rate:.1f}%", "kpi-teal"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 2: Sentiment Donut + Trend ---
        col_left, col_right = st.columns([1, 2])

        with col_left:
            section_header("Sentiment Distribution")
            sent_counts = df["sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            colors = [SENTIMENT_COLORS.get(s, COLORS["slate"]) for s in sent_counts.index]
            wedges, texts, autotexts = ax.pie(
                sent_counts.values,
                labels=sent_counts.index,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.78,
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
            )
            for t in autotexts:
                t.set_fontsize(11)
                t.set_fontweight("bold")
                t.set_color("white")
            for t in texts:
                t.set_fontsize(10)
                t.set_color(COLORS["navy"])
            centre_circle = plt.Circle((0, 0), 0.55, fc="white")
            ax.add_artist(centre_circle)
            ax.set_title("")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_right:
            section_header("Negative Feedback Trend (Monthly)")
            df_trend = df.copy()
            df_trend["month"] = df_trend["feedback_date"].dt.to_period("M")
            monthly = df_trend.groupby("month").agg(
                total=("sentiment", "count"),
                negative=("sentiment", lambda x: (x == "Negative").sum()),
            ).reset_index()
            monthly["month_str"] = monthly["month"].astype(str)
            monthly["neg_rate"] = (monthly["negative"] / monthly["total"] * 100).round(1)

            fig, ax1 = plt.subplots(figsize=(10, 4.5))
            x = range(len(monthly))
            ax1.bar(x, monthly["total"], color=COLORS["light_teal"], alpha=0.5, label="Total Feedback", width=0.6)
            ax1.set_ylabel("Total Feedback", color=COLORS["slate"])
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
            style_chart(ax1)

            ax2 = ax1.twinx()
            ax2.plot(x, monthly["neg_rate"], color=COLORS["red"], marker="o", linewidth=2.5,
                     markersize=6, label="Negative Rate %", zorder=5)
            ax2.set_ylabel("Negative Rate (%)", color=COLORS["red"])
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color(COLORS["red"])
            ax2.tick_params(axis="y", colors=COLORS["red"])

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.9)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # --- Row 3: Priority Distribution + Source Breakdown ---
        col_a, col_b = st.columns(2)

        with col_a:
            section_header("Priority Distribution")
            pri_counts = df["priority"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.barh(pri_counts.index[::-1], pri_counts.values[::-1],
                          color=[PRIORITY_COLORS[p] for p in pri_counts.index[::-1]],
                          height=0.5, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, pri_counts.values[::-1]):
                ax.text(bar.get_width() + max(pri_counts.values)*0.02, bar.get_y() + bar.get_height()/2,
                       f"{val:,}", va="center", fontsize=11, fontweight="bold", color=COLORS["navy"])
            style_chart(ax, xlabel="Number of Records")
            ax.set_xlim(0, max(pri_counts.values) * 1.15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            section_header("Feedback by Source")
            source_counts = df["source"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(range(len(source_counts)), source_counts.values,
                         color=CHART_PALETTE[:len(source_counts)], width=0.6, edgecolor="white", linewidth=1)
            ax.set_xticks(range(len(source_counts)))
            ax.set_xticklabels(source_counts.index, rotation=40, ha="right", fontsize=8)
            for bar, val in zip(bars, source_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(source_counts.values)*0.02,
                       str(val), ha="center", fontsize=9, fontweight="bold", color=COLORS["navy"])
            style_chart(ax, ylabel="Count")
            ax.set_ylim(0, max(source_counts.values) * 1.18)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ===========================================================================
# PAGE 2: COMPLAINT INTELLIGENCE
# ===========================================================================

elif page == "Complaint Intelligence":
    st.markdown(
        '<h1 style="color:#0f2b5b; font-weight:700; margin-bottom:4px;">'
        '&#128200; Complaint Intelligence</h1>'
        '<p style="color:#64748b; margin-top:0;">Drill into complaint categories, route hotspots, and airline comparisons</p>',
        unsafe_allow_html=True,
    )

    if len(df) == 0:
        st.info("No data available.")
    else:
        # --- Complaint Category Volume ---
        section_header("Complaint Volume by Category")
        cat_counts = df["issue_category"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
                      color=CHART_PALETTE[:len(cat_counts)], height=0.55, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, cat_counts.values[::-1]):
            ax.text(bar.get_width() + max(cat_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:,}", va="center", fontsize=10, fontweight="bold", color=COLORS["navy"])
        style_chart(ax, xlabel="Number of Records")
        ax.set_xlim(0, max(cat_counts.values) * 1.12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row: Negative Rate by Category + Status Distribution ---
        col1, col2 = st.columns(2)

        with col1:
            section_header("Negative Feedback Rate by Category")
            cat_neg = df.groupby("issue_category").agg(
                total=("sentiment", "count"),
                negative=("sentiment", lambda x: (x == "Negative").sum()),
            )
            cat_neg["neg_rate"] = (cat_neg["negative"] / cat_neg["total"] * 100).round(1)
            cat_neg = cat_neg.sort_values("neg_rate", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            bar_colors = [COLORS["red"] if r > 50 else COLORS["amber"] if r > 30 else COLORS["teal"]
                         for r in cat_neg["neg_rate"]]
            bars = ax.barh(cat_neg.index, cat_neg["neg_rate"],
                          color=bar_colors, height=0.5, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, cat_neg["neg_rate"]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color=COLORS["navy"])
            style_chart(ax, xlabel="Negative Rate (%)")
            ax.set_xlim(0, max(cat_neg["neg_rate"]) * 1.15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            section_header("Case Status Distribution")
            status_counts = df["case_status"].value_counts()
            status_colors = [COLORS["red"], COLORS["amber"], COLORS["teal"], COLORS["navy"]]
            fig, ax = plt.subplots(figsize=(5, 4.5))
            wedges, texts, autotexts = ax.pie(
                status_counts.values,
                labels=status_counts.index,
                colors=status_colors[:len(status_counts)],
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.8,
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
            )
            for t in autotexts:
                t.set_fontsize(10)
                t.set_fontweight("bold")
                t.set_color("white")
            for t in texts:
                t.set_fontsize(9)
                t.set_color(COLORS["navy"])
            centre = plt.Circle((0, 0), 0.5, fc="white")
            ax.add_artist(centre)
            fig.patch.set_facecolor("white")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Route Hotspots ---
        section_header("Top 10 Routes by Negative Feedback")
        route_neg = df[df["sentiment"] == "Negative"]["route"].value_counts().head(10)
        if len(route_neg) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(len(route_neg)), route_neg.values,
                         color=COLORS["red"], alpha=0.85, width=0.6, edgecolor="white", linewidth=1)
            ax.set_xticks(range(len(route_neg)))
            ax.set_xticklabels(route_neg.index, rotation=45, ha="right", fontsize=9)
            for bar, val in zip(bars, route_neg.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       str(val), ha="center", fontsize=9, fontweight="bold", color=COLORS["navy"])
            style_chart(ax, ylabel="Negative Reviews")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Airline Comparison (only shown when "All Airlines" is selected) ---
        if selected_airline == "All Airlines":
            section_header("Airline Sentiment Comparison")
            airline_sent = df.groupby(["airline", "sentiment"]).size().unstack(fill_value=0)
            # Ensure all sentiment columns exist
            for s in ["Positive", "Negative", "Neutral"]:
                if s not in airline_sent.columns:
                    airline_sent[s] = 0
            airline_sent = airline_sent[["Positive", "Neutral", "Negative"]]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = range(len(airline_sent))
            width = 0.25
            ax.bar([i - width for i in x], airline_sent["Positive"], width=width,
                   color=COLORS["teal"], label="Positive", edgecolor="white", linewidth=1)
            ax.bar(x, airline_sent["Neutral"], width=width,
                   color=COLORS["slate"], label="Neutral", edgecolor="white", linewidth=1)
            ax.bar([i + width for i in x], airline_sent["Negative"], width=width,
                   color=COLORS["red"], label="Negative", edgecolor="white", linewidth=1)
            ax.set_xticks(list(x))
            ax.set_xticklabels(airline_sent.index, fontsize=10)
            ax.legend(fontsize=9, framealpha=0.9)
            style_chart(ax, ylabel="Number of Reviews")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown(
                '<div class="info-box">'
                '<b>Note:</b> This comparison uses synthetic data distributed across airlines. '
                'These figures do not represent actual airline performance and should not be '
                'used for benchmarking.'
                '</div>',
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE 3: FEEDBACK EXPLORER
# ===========================================================================

elif page == "Feedback Explorer":
    st.markdown(
        '<h1 style="color:#0f2b5b; font-weight:700; margin-bottom:4px;">'
        '&#128269; Feedback Explorer</h1>'
        '<p style="color:#64748b; margin-top:0;">Search, filter, and inspect individual feedback records</p>',
        unsafe_allow_html=True,
    )

    if len(df) == 0:
        st.info("No data available.")
    else:
        # --- Filters ---
        section_header("Filters")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            sent_filter = st.multiselect(
                "Sentiment", options=sorted(df["sentiment"].unique()), default=sorted(df["sentiment"].unique())
            )
        with fc2:
            cat_filter = st.multiselect(
                "Category", options=sorted(df["issue_category"].unique()), default=sorted(df["issue_category"].unique())
            )
        with fc3:
            pri_filter = st.multiselect(
                "Priority", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"]
            )
        with fc4:
            source_filter = st.multiselect(
                "Source", options=sorted(df["source"].unique()), default=sorted(df["source"].unique())
            )

        # Search bar
        search_term = st.text_input("Search in review text", placeholder="e.g. refund, delay, baggage...")

        # Apply filters
        df_filtered = df[
            (df["sentiment"].isin(sent_filter))
            & (df["issue_category"].isin(cat_filter))
            & (df["priority"].isin(pri_filter))
            & (df["source"].isin(source_filter))
        ]

        if search_term.strip():
            df_filtered = df_filtered[
                df_filtered["review_text"].str.contains(search_term.strip(), case=False, na=False)
            ]

        st.markdown(f"**{len(df_filtered):,}** records match your filters", unsafe_allow_html=True)

        # --- Display table ---
        display_cols = [
            "feedback_id", "feedback_date", "airline", "route", "source",
            "review_text", "sentiment", "issue_category", "priority",
            "satisfaction_rating", "case_status",
        ]
        available_cols = [c for c in display_cols if c in df_filtered.columns]

        st.dataframe(
            df_filtered[available_cols].head(200),
            use_container_width=True,
            height=450,
        )

        # --- Export ---
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = df_filtered[available_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"aerolens_feedback_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # --- CSV Template Download ---
        st.markdown("---")
        st.download_button(
            label="Download CSV Upload Template",
            data=get_template_csv(),
            file_name="aerolens_upload_template.csv",
            mime="text/csv",
        )


# ===========================================================================
# PAGE 4: FEEDBACK ANALYSER
# ===========================================================================

elif page == "Feedback Analyser":
    st.markdown(
        '<h1 style="color:#0f2b5b; font-weight:700; margin-bottom:4px;">'
        '&#128300; Feedback Analyser</h1>'
        '<p style="color:#64748b; margin-top:0;">Analyse individual reviews or upload a CSV for batch classification</p>',
        unsafe_allow_html=True,
    )

    try:
        models = get_models()
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"Models not found. Run train.py first.\n\n{e}")
        models_loaded = False

    if models_loaded:
        tab1, tab2, tab3 = st.tabs([
            "Single Review Analysis",
            "Batch CSV Upload",
            "Model Performance",
        ])

        # --- Tab 1: Single Review ---
        with tab1:
            section_header("Analyse a Single Review")

            review_input = st.text_area(
                "Enter customer feedback",
                height=120,
                placeholder="e.g. My flight was delayed by 4 hours and no one provided any information. The ground staff was unhelpful.",
            )

            if st.button("Analyse Review", type="primary"):
                if review_input.strip():
                    result = predict_single(review_input, models)

                    st.markdown("<br>", unsafe_allow_html=True)
                    rc1, rc2, rc3 = st.columns(3)

                    with rc1:
                        sent_color = {"Positive": "kpi-teal", "Negative": "kpi-red", "Neutral": "kpi-navy"}.get(
                            result["sentiment"], "kpi-navy"
                        )
                        st.markdown(
                            kpi_card("Predicted Sentiment", result["sentiment"], sent_color),
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Confidence: {result['sentiment_confidence']:.1%}")

                    with rc2:
                        st.markdown(
                            kpi_card("Predicted Category", result["category"], "kpi-navy"),
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Confidence: {result['category_confidence']:.1%}")

                    with rc3:
                        pri_color = {"High": "kpi-red", "Medium": "kpi-amber", "Low": "kpi-teal"}.get(
                            result["priority"], "kpi-navy"
                        )
                        st.markdown(
                            kpi_card("Priority (Rule-Based)", result["priority"], pri_color),
                            unsafe_allow_html=True,
                        )
                        st.caption("Based on sentiment + category business rules")

                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("View cleaned text"):
                        st.code(result["cleaned_text"])
                else:
                    st.warning("Please enter a review to analyse.")

        # --- Tab 2: Batch Upload ---
        with tab2:
            section_header("Batch CSV Classification")

            st.markdown(
                '<div class="info-box">'
                'Upload a CSV file with a <b>review_text</b> column. '
                'The system will predict sentiment, category, and priority for each row. '
                'Additional columns (airline, route, source, etc.) will be preserved.'
                '</div>',
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df_upload):,} records")

                    if "review_text" not in df_upload.columns:
                        st.error("CSV must contain a 'review_text' column.")
                    else:
                        # Preview raw data
                        with st.expander("Preview uploaded data"):
                            st.dataframe(df_upload.head(20), use_container_width=True)

                        if st.button("Run Batch Analysis", type="primary"):
                            with st.spinner("Classifying reviews..."):
                                df_result = predict_batch(df_upload, models, text_column="review_text")

                            st.success(f"Analysis complete for {len(df_result):,} reviews!")

                            # Summary stats
                            sc1, sc2, sc3 = st.columns(3)
                            with sc1:
                                st.metric("Positive", f"{(df_result['predicted_sentiment'] == 'Positive').sum()}")
                            with sc2:
                                st.metric("Neutral", f"{(df_result['predicted_sentiment'] == 'Neutral').sum()}")
                            with sc3:
                                st.metric("Negative", f"{(df_result['predicted_sentiment'] == 'Negative').sum()}")

                            # Show results
                            st.dataframe(df_result.head(100), use_container_width=True, height=400)

                            # Download
                            csv_result = df_result.to_csv(index=False)
                            st.download_button(
                                label="Download Analysed Data (CSV)",
                                data=csv_result,
                                file_name=f"aerolens_batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                            )

                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        # --- Tab 3: Model Performance ---
        with tab3:
            section_header("Model Evaluation Metrics")

            st.markdown(
                '<div class="info-box">'
                '<b>Important:</b> These metrics are computed on synthetic (computer-generated) data. '
                'They do not represent real-world model accuracy. Models would require retraining '
                'on genuine customer feedback before production deployment.'
                '</div>',
                unsafe_allow_html=True,
            )

            eval_results = load_eval_results()

            if eval_results:
                for model_key, label in [("sentiment", "Sentiment Classifier"), ("category", "Category Classifier")]:
                    if model_key in eval_results:
                        res = eval_results[model_key]
                        st.markdown(f"#### {label}")

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Accuracy", f"{res['accuracy']:.2%}")
                        mc2.metric("Precision (Macro)", f"{res['precision_macro']:.2%}")
                        mc3.metric("Recall (Macro)", f"{res['recall_macro']:.2%}")
                        mc4.metric("F1 Weighted", f"{res['f1_weighted']:.2%}")

                        # Confusion matrix
                        with st.expander(f"View Confusion Matrix - {label}"):
                            cm = np.array(res["confusion_matrix"])
                            labels = res["labels"]
                            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                            cm_df.index.name = "Actual"
                            cm_df.columns.name = "Predicted"
                            st.dataframe(cm_df, use_container_width=True)

                        st.markdown("---")
            else:
                st.warning("Evaluation results not found. Run evaluate.py first.")
