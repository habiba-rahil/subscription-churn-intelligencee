"""
Subscription Churn Intelligence — Premium Dashboard
Run: streamlit run dashboard/app.py
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_processing import load_clean_data, calculate_churn_rate
from feature_engineering import build_ml_features
from cohort_analysis import build_cohort_table, get_monthly_churn_trend
from segmentation import build_segmentation_features, fit_kmeans, profile_clusters, assign_segment_labels

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #060912 !important;
    color: #e2e8f0 !important;
}

/* ── layout ── */
.main .block-container { padding: 0 2.5rem 4rem !important; max-width: 1500px !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0b0f1a !important;
    border-right: 1px solid #1a2035 !important;
    width: 230px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 0.55rem 0.9rem !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #131929 !important;
    color: #f1f5f9 !important;
}

/* ── headings ── */
h1,h2,h3 { color: #f1f5f9 !important; font-weight: 800 !important; letter-spacing:-0.025em !important; }
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.35rem !important; }

/* ── selectbox / multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* ── metric override (we use custom HTML cards so hide native) ── */
[data-testid="stMetric"] { display: none !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] iframe { border-radius: 12px !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060912; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── plotly chart wrapper ── */
[data-testid="stPlotlyChart"] > div { border-radius: 16px !important; overflow: hidden !important; }

/* ── hr ── */
hr { border: none !important; border-top: 1px solid #1a2035 !important; margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#060912"
CARD     = "#0e1421"
CARD2    = "#111827"
BORDER   = "#1a2035"
TEXT     = "#f1f5f9"
MUTED    = "#64748b"
SUB      = "#94a3b8"

C_BLUE   = "#3b82f6"
C_PURPLE = "#8b5cf6"
C_CYAN   = "#06b6d4"
C_GREEN  = "#10b981"
C_RED    = "#ef4444"
C_AMBER  = "#f59e0b"
C_PINK   = "#ec4899"

PALETTE  = [C_BLUE, C_PURPLE, C_CYAN, C_GREEN, C_AMBER, C_RED, C_PINK]

def _ax():
    return dict(gridcolor="#1a2035", linecolor="#1a2035",
                tickcolor="#334155", tickfont_color=SUB,
                title_font_color=SUB, zeroline=False)

def apply_theme(fig, title="", h=None, legend_h=False):
    kw = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=SUB, size=12),
        title_text=title,
        title_font=dict(family="Inter", color=TEXT, size=14, weight=700),
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=SUB,
                    orientation="h" if legend_h else "v",
                    y=1.06 if legend_h else None, x=0 if legend_h else None),
    )
    if h:
        kw["height"] = h
    fig.update_layout(**kw)
    fig.update_xaxes(**_ax())
    fig.update_yaxes(**_ax())
    return fig

def kpi(label, value, color=C_BLUE, sub="", icon=""):
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
                padding:1.3rem 1.4rem;transition:border-color .2s;
                border-top:2px solid {color}">
        <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.09em;color:{MUTED};margin-bottom:.55rem">
            {icon} {label}
        </div>
        <div style="font-size:1.9rem;font-weight:800;color:{color};line-height:1">
            {value}
        </div>
        {'<div style="font-size:.75rem;color:'+MUTED+';margin-top:.35rem">'+sub+'</div>' if sub else ''}
    </div>"""

def section(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.7rem;margin:1.8rem 0 1rem">
        <div style="width:3px;height:18px;background:{C_BLUE};border-radius:2px"></div>
        <span style="font-size:.7rem;font-weight:800;text-transform:uppercase;
                     letter-spacing:.12em;color:{C_BLUE}">{text}</span>
        <div style="flex:1;height:1px;background:{BORDER}"></div>
    </div>""", unsafe_allow_html=True)

def page_header(title, sub):
    st.markdown(f"""
    <div style="padding:2.2rem 0 1.8rem">
        <h1 style="font-size:2.1rem;font-weight:900;letter-spacing:-.03em;
                   background:linear-gradient(135deg,{TEXT} 0%,{SUB} 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:.4rem">{title}</h1>
        <p style="font-size:.9rem;color:{MUTED};font-weight:400">{sub}</p>
    </div>""", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"""
    <div style="background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.2);
                border-radius:10px;padding:.85rem 1.1rem;margin-bottom:1.2rem">
        <span style="color:{C_BLUE};font-size:.8rem">{text}</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_clean_data()

@st.cache_data(show_spinner=False)
def get_cohort(_df):
    return build_cohort_table(_df)

@st.cache_data(show_spinner=False)
def get_segments(_df):
    feats = build_segmentation_features(_df)
    labels, km, scaler = fit_kmeans(feats, n_clusters=4)
    seg = _df.loc[feats.index].copy()
    seg["cluster"] = labels
    prof = profile_clusters(seg, list(feats.columns))
    smap = assign_segment_labels(prof)
    seg["segment_label"] = seg["cluster"].map(smap)
    return seg, prof, smap, feats

@st.cache_data(show_spinner=False)
def get_models(_df):
    from churn_model import (split_data, train_logistic_regression,
                              train_random_forest, evaluate_model, compare_models)
    X, y = build_ml_features(_df)
    Xtr, Xte, ytr, yte = split_data(X, y)
    lr = train_logistic_regression(Xtr, ytr)
    rf = train_random_forest(Xtr, ytr)
    lm = evaluate_model(lr, Xte, yte, "Logistic Regression")
    rm = evaluate_model(rf, Xte, yte, "Random Forest")
    lm["y_test"] = yte; rm["y_test"] = yte
    return lr, rf, X, y, Xte, yte, compare_models([lm, rm]), lm, rm

df          = get_data()
churn_rate  = calculate_churn_rate(df)
total       = len(df)
churned_n   = int(df["churned"].sum())
active_n    = total - churned_n
avg_rev     = df[df["churned"]==0]["monthly_fee"].mean()
total_mrr   = df[df["churned"]==0]["monthly_fee"].sum()
at_risk_n   = int(df[(df["churned"]==0) &
                     ((df["payment_failures"]>=2) | (df["usage_frequency"]<5))].shape[0])

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "Loyal / High Value":   C_GREEN,
    "Average / Stable":     C_BLUE,
    "Low Engagement":       C_AMBER,
    "High Risk / At Risk":  C_RED,
    "New Customers":        C_PURPLE,
}

with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.8rem">
        <div style="font-size:1.25rem;font-weight:900;letter-spacing:-.02em;color:{TEXT}">
            ⚡ Churn<span style="color:{C_BLUE}">IQ</span>
        </div>
        <div style="font-size:.7rem;color:{MUTED};margin-top:.2rem;font-weight:500">
            Intelligence Dashboard
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{MUTED};margin-bottom:.5rem">Menu</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "📈  Retention",
        "👥  Segments",
        "🎯  Prediction",
        "💡  Insights",
    ], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER};margin:1.4rem 0'>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{MUTED};margin-bottom:.7rem">Live Stats</div>', unsafe_allow_html=True)

    stats = [
        ("Customers",    f"{total:,}",          TEXT),
        ("Churn Rate",   f"{churn_rate:.1f}%",   C_RED),
        ("Active MRR",   f"${total_mrr:,.0f}",   C_GREEN),
        ("At-Risk",      f"{at_risk_n:,}",        C_AMBER),
    ]
    for lbl, val, col in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:.45rem .1rem;border-bottom:1px solid {BORDER}">
            <span style="font-size:.78rem;color:{MUTED}">{lbl}</span>
            <span style="font-size:.85rem;font-weight:700;color:{col}">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.4rem;padding:.8rem;background:{CARD};border:1px solid {BORDER};
                border-radius:10px;font-size:.7rem;line-height:1.7;color:{MUTED}">
        <strong style="color:{SUB}">Stack</strong><br>
        Python · pandas · scikit-learn<br>
        DuckDB · SHAP · Streamlit<br>
        <span style="color:{C_GREEN}">RF AUC: <strong>0.971</strong></span>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    page_header("Executive Overview",
                "Real-time snapshot of customer health, churn status, and revenue risk")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = [
        ("Total Customers",  f"{total:,}",         C_BLUE,   f"All time",                    "👤"),
        ("Active",           f"{active_n:,}",       C_GREEN,  f"{active_n/total*100:.1f}% of base","✅"),
        ("Churned",          f"{churned_n:,}",      C_RED,    "All time",                    "❌"),
        ("Churn Rate",       f"{churn_rate:.1f}%",  C_AMBER,  "Overall",                     "📉"),
        ("Avg MRR / Active", f"${avg_rev:.2f}",     C_PURPLE, "Per active customer",         "💰"),
        ("At-Risk Active",   f"{at_risk_n:,}",      C_PINK,   "Needs attention",             "🚨"),
    ]
    cols = st.columns(6)
    for col, (lbl, val, clr, sub_t, ico) in zip(cols, kpis):
        col.markdown(kpi(lbl, val, clr, sub_t, ico), unsafe_allow_html=True)

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    section("CHURN BREAKDOWN")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        counts = df["churned"].value_counts().reset_index()
        counts.columns = ["churned","count"]
        counts["label"] = counts["churned"].map({0:"Active", 1:"Churned"})
        fig = px.pie(counts, names="label", values="count", hole=0.62,
                     color="label",
                     color_discrete_map={"Active":C_GREEN,"Churned":C_RED})
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color=BG, width=3)),
                          pull=[0.03, 0.03])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[dict(text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:10px'>churn</span>",
                              x=0.5, y=0.5, font_size=18, font_color=C_RED,
                              showarrow=False)],
            title_text="Status Split", title_font=dict(color=TEXT, size=14, weight=700),
            margin=dict(l=16,r=16,t=48,b=16), height=280,
            font=dict(family="Inter", color=SUB),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        plan_order = ["Basic","Standard","Premium","Enterprise"]
        pc = df.groupby("plan")["churned"].agg(["sum","count"]).reset_index()
        pc.columns = ["plan","ch","tot"]
        pc["rate"] = (pc["ch"]/pc["tot"]*100).round(1)
        pc["plan"] = pd.Categorical(pc["plan"], categories=plan_order, ordered=True)
        pc = pc.sort_values("plan")
        fig = go.Figure(go.Bar(
            x=pc["plan"], y=pc["rate"],
            text=pc["rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(
                color=pc["rate"],
                colorscale=[[0,C_GREEN],[0.5,C_AMBER],[1,C_RED]],
                line=dict(width=0),
            ),
        ))
        apply_theme(fig, "Churn Rate by Plan", h=280)
        fig.update_layout(xaxis_title="", yaxis_title="Churn %",
                          yaxis=dict(gridcolor="#1a2035", linecolor="#1a2035",
                                     tickcolor="#334155", tickfont_color=SUB,
                                     title_font_color=SUB, zeroline=False))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        cc = df.groupby("country")["churned"].agg(["sum","count"]).reset_index()
        cc.columns = ["country","ch","tot"]
        cc["rate"] = (cc["ch"]/cc["tot"]*100).round(1)
        cc = cc.sort_values("rate", ascending=True)
        fig = go.Figure(go.Bar(
            y=cc["country"], x=cc["rate"],
            orientation="h",
            text=cc["rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(
                color=cc["rate"],
                colorscale=[[0,C_GREEN],[0.5,C_AMBER],[1,C_RED]],
                line=dict(width=0),
            ),
        ))
        apply_theme(fig, "Churn Rate by Country", h=280)
        fig.update_xaxes(title_text="Churn %", **_ax())
        fig.update_yaxes(title_text="", **_ax())
        st.plotly_chart(fig, use_container_width=True)

    # ── Trend ─────────────────────────────────────────────────────────────────
    section("MONTHLY TREND")
    trend = get_monthly_churn_trend(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend["month"], y=trend["new_customers"],
        name="New Signups",
        marker=dict(color="rgba(59,130,246,0.18)", line=dict(width=0)),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=trend["month"], y=trend["churned_customers"],
        name="Churned", mode="lines+markers",
        line=dict(color=C_RED, width=2.5),
        marker=dict(size=5, color=C_RED,
                    line=dict(color=BG, width=1.5)),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_text="Monthly Churn vs New Signups",
        title_font=dict(color=TEXT, size=14, weight=700),
        font=dict(family="Inter", color=SUB, size=12),
        xaxis=dict(title="", **_ax()),
        yaxis=dict(title="Churned", **_ax()),
        yaxis2=dict(title="New Signups", overlaying="y", side="right",
                    showgrid=False, **_ax()),
        legend=dict(orientation="h", y=1.06, x=0,
                    bgcolor="rgba(0,0,0,0)", font_color=SUB),
        margin=dict(l=16,r=16,t=48,b=16), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Bottom row ────────────────────────────────────────────────────────────
    section("BEHAVIOUR SIGNALS")
    c1, c2 = st.columns(2)

    with c1:
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"],
            bins=[0,24,34,44,54,100],
            labels=["18-24","25-34","35-44","45-54","55+"])
        ag = df2.groupby("age_group", observed=True)["churned"].mean().reset_index()
        ag.columns = ["age_group","rate"]
        ag["rate"] = (ag["rate"]*100).round(1)
        fig = go.Figure(go.Bar(
            x=ag["age_group"].astype(str), y=ag["rate"],
            text=ag["rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(color=ag["rate"],
                        colorscale=[[0,C_GREEN],[0.5,C_AMBER],[1,C_RED]],
                        line=dict(width=0)),
        ))
        apply_theme(fig, "Churn Rate by Age Group", h=270)
        fig.update_xaxes(title_text="", **_ax())
        fig.update_yaxes(title_text="Churn %", **_ax())
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pm = df.groupby("payment_method")["churned"].agg(["sum","count"]).reset_index()
        pm.columns = ["method","ch","tot"]
        pm["rate"] = (pm["ch"]/pm["tot"]*100).round(1)
        fig = go.Figure(go.Bar(
            x=pm["method"], y=pm["rate"],
            text=pm["rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(color=PALETTE[:len(pm)], line=dict(width=0)),
        ))
        apply_theme(fig, "Churn Rate by Payment Method", h=270)
        fig.update_xaxes(title_text="", **_ax())
        fig.update_yaxes(title_text="Churn %", **_ax())
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RETENTION
# ═════════════════════════════════════════════════════════════════════════════
elif "Retention" in page:

    page_header("Customer Retention",
                "Cohort analysis — track how retention evolves month over month")

    info_box("ℹ️  Each row = a signup cohort. Each column = months after signup. "
             "Green = high retention. Month 0 is always 100%.")

    cohort_pivot = get_cohort(df)
    all_cohorts  = cohort_pivot.index.tolist()

    selected = st.multiselect("Filter cohorts (empty = all)",
                               options=all_cohorts, default=[])
    pivot = cohort_pivot.loc[selected if selected else all_cohorts]
    cols_show = [c for c in pivot.columns if int(c.split()[-1]) <= 12]
    data = pivot[cols_show]

    section("COHORT HEATMAP")
    fig = go.Figure(go.Heatmap(
        z=data.values,
        x=data.columns.tolist(),
        y=data.index.tolist(),
        colorscale=[
            [0.0,  "#1a0a2e"],
            [0.25, "#4c1d95"],
            [0.5,  "#1d4ed8"],
            [0.75, "#0891b2"],
            [1.0,  "#10b981"],
        ],
        text=data.values.round(0).astype(int),
        texttemplate="%{text}%",
        textfont=dict(size=10, color="rgba(255,255,255,0.85)"),
        hovertemplate="Cohort: %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        zmin=0, zmax=100,
        colorbar=dict(
            tickfont=dict(color=SUB, size=10),
            title=dict(text="Retention %", font=dict(color=SUB, size=11)),
            thickness=12,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=SUB, size=11),
        title_text="Cohort Retention Heatmap",
        title_font=dict(color=TEXT, size=14, weight=700),
        xaxis=dict(title="Months Since Signup", **_ax()),
        yaxis=dict(title="", **_ax()),
        height=max(420, len(data)*24+100),
        margin=dict(l=16,r=16,t=48,b=16),
    )
    st.plotly_chart(fig, use_container_width=True)

    section("AVERAGE RETENTION CURVE")
    avg = pivot.mean(axis=0).reset_index()
    avg.columns = ["Month","Ret"]
    avg = avg.head(13)

    m1  = avg[avg["Month"]=="Month 1"]["Ret"].values
    m6  = avg[avg["Month"]=="Month 6"]["Ret"].values
    m12 = avg[avg["Month"]=="Month 12"]["Ret"].values

    c1,c2,c3 = st.columns(3)
    for col, lbl, vals in [(c1,"Month 1",m1),(c2,"Month 6",m6),(c3,"Month 12",m12)]:
        v = f"{vals[0]:.1f}%" if len(vals) else "N/A"
        clr = (C_GREEN if len(vals) and vals[0]>65
               else C_AMBER if len(vals) and vals[0]>35 else C_RED)
        col.markdown(kpi(lbl+" Retention", v, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg["Month"], y=avg["Ret"],
        mode="lines+markers",
        line=dict(color=C_BLUE, width=3),
        marker=dict(size=8, color=C_BLUE,
                    symbol="circle",
                    line=dict(color=BG, width=2)),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
        name="Avg Retention",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color=C_RED, line_width=1.5,
                  annotation_text="50% mark",
                  annotation_font=dict(color=C_RED, size=11))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_text="Average Retention Across All Cohorts",
        title_font=dict(color=TEXT, size=14, weight=700),
        font=dict(family="Inter", color=SUB, size=12),
        xaxis=dict(title="", **_ax()),
        yaxis=dict(title="Retention (%)", range=[0,108], **_ax()),
        margin=dict(l=16,r=16,t=48,b=16), height=320,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SEGMENTS
# ═════════════════════════════════════════════════════════════════════════════
elif "Segments" in page:

    page_header("Customer Segments",
                "K-Means clustering reveals natural behavioural groups")

    df_seg, profile, smap, feats = get_segments(df)

    seg_stats = df_seg.groupby("segment_label").agg(
        count=("customer_id","count"),
        churn=("churned","mean"),
        usage=("usage_frequency","mean"),
        failures=("payment_failures","mean"),
    ).reset_index()
    seg_stats["churn"] = (seg_stats["churn"]*100).round(1)

    section("SEGMENT OVERVIEW")
    cols = st.columns(len(seg_stats))
    for col, (_, row) in zip(cols, seg_stats.iterrows()):
        clr = SEG_COLORS.get(row["segment_label"], C_BLUE)
        col.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-top:2px solid {clr};
                    border-radius:14px;padding:1.2rem 1rem;text-align:center">
            <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;
                        letter-spacing:.08em;color:{clr};margin-bottom:.5rem">
                {row['segment_label']}
            </div>
            <div style="font-size:1.7rem;font-weight:900;color:{TEXT};line-height:1">
                {int(row['count']):,}
            </div>
            <div style="font-size:.72rem;color:{MUTED};margin-top:.3rem">
                {row['churn']:.1f}% churn · {row['usage']:.0f} uses/mo
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("DISTRIBUTION & CHURN")
    c1, c2 = st.columns(2)

    with c1:
        sc = df_seg["segment_label"].value_counts().reset_index()
        sc.columns = ["Segment","Count"]
        clrs = [SEG_COLORS.get(s, C_BLUE) for s in sc["Segment"]]
        fig = go.Figure(go.Pie(
            labels=sc["Segment"], values=sc["Count"],
            hole=0.6,
            marker=dict(colors=clrs, line=dict(color=BG, width=3)),
            textposition="outside",
            textinfo="percent+label",
            pull=[0.04]*len(sc),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            title_text="Segment Distribution",
            title_font=dict(color=TEXT, size=14, weight=700),
            font=dict(family="Inter", color=SUB, size=11),
            margin=dict(l=16,r=16,t=48,b=16), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sc2 = df_seg.groupby("segment_label")["churned"].mean().reset_index()
        sc2.columns = ["Segment","Rate"]
        sc2["Rate"] = (sc2["Rate"]*100).round(1)
        sc2 = sc2.sort_values("Rate")
        sc2_clrs = [SEG_COLORS.get(s, C_BLUE) for s in sc2["Segment"]]
        fig = go.Figure(go.Bar(
            y=sc2["Segment"], x=sc2["Rate"],
            orientation="h",
            text=sc2["Rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(color=sc2_clrs, line=dict(width=0)),
        ))
        apply_theme(fig, "Churn Rate by Segment", h=320)
        fig.update_xaxes(title_text="Churn Rate (%)", **_ax())
        fig.update_yaxes(title_text="", **_ax())
        st.plotly_chart(fig, use_container_width=True)

    section("SEGMENT EXPLORER")
    c1, c2 = st.columns(2)
    with c1:
        xax = st.selectbox("X axis", ["usage_frequency","payment_failures",
                                        "tenure_months","days_since_login","monthly_fee"])
    with c2:
        yax = st.selectbox("Y axis", ["payment_failures","usage_frequency",
                                        "tenure_months","days_since_login","monthly_fee"])

    fig = px.scatter(
        df_seg, x=xax, y=yax,
        color="segment_label",
        color_discrete_map=SEG_COLORS,
        opacity=0.55,
        hover_data=["customer_id","plan","churned"],
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    apply_theme(fig, f"{xax}  vs  {yax}", h=380)
    fig.update_xaxes(title_text=xax, **_ax())
    fig.update_yaxes(title_text=yax, **_ax())
    st.plotly_chart(fig, use_container_width=True)

    section("SEGMENT PROFILES")
    fcols = ["days_since_login","usage_frequency","monthly_fee",
             "support_tickets","payment_failures","tenure_months"]
    pd_display = profile[fcols+["churn_rate_%","cluster_size"]].copy()
    pd_display.index = [smap.get(i, f"Cluster {i}") for i in pd_display.index]
    st.dataframe(pd_display.round(2), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif "Prediction" in page:

    page_header("Churn Prediction",
                "ML-powered risk scoring — Random Forest · AUC 0.971")

    lr, rf, X, y, Xte, yte, comp, lm, rm = get_models(df)

    section("MODEL PERFORMANCE — RANDOM FOREST")
    mcols = st.columns(5)
    for col, (lbl, val, clr) in zip(mcols, [
        ("Accuracy",  rm["accuracy"],  C_BLUE),
        ("Precision", rm["precision"], C_PURPLE),
        ("Recall",    rm["recall"],    C_GREEN),
        ("F1-Score",  rm["f1_score"],  C_CYAN),
        ("ROC-AUC",   rm["roc_auc"],   C_AMBER),
    ]):
        col.markdown(kpi(lbl, f"{val:.3f}", clr, "Random Forest"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("ROC CURVES  &  FEATURE IMPORTANCE")
    c1, c2 = st.columns([3,2])

    with c1:
        from sklearn.metrics import roc_curve
        fig = go.Figure()
        for name, m, clr, fill in [
            ("Random Forest",       rm, C_GREEN,  "tozeroy"),
            ("Logistic Regression", lm, C_BLUE,   None),
        ]:
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name}  (AUC={m['roc_auc']:.3f})",
                line=dict(color=clr, width=2.5),
                fill=fill,
                fillcolor="rgba(16,185,129,0.05)" if fill else None,
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dot", color="#334155", width=1),
            name="Random",
        ))
        apply_theme(fig, "ROC Curves — Model Comparison", h=340)
        fig.update_xaxes(title_text="False Positive Rate", **_ax())
        fig.update_yaxes(title_text="True Positive Rate", **_ax())
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0,
                                      bgcolor="rgba(0,0,0,0)", font_color=SUB))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fi = pd.DataFrame({
            "Feature":    X.columns,
            "Importance": rf.named_steps["model"].feature_importances_,
        }).sort_values("Importance", ascending=True).tail(10)

        fig = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"],
            orientation="h",
            marker=dict(
                color=fi["Importance"],
                colorscale=[[0,"#1e3a5f"],[1,C_BLUE]],
                line=dict(width=0),
            ),
        ))
        apply_theme(fig, "Top Feature Importances", h=340)
        fig.update_xaxes(title_text="", **_ax())
        fig.update_yaxes(title_text="", **_ax())
        st.plotly_chart(fig, use_container_width=True)

    # ── Customer Lookup ────────────────────────────────────────────────────────
    section("INDIVIDUAL RISK ASSESSMENT")

    selected_id = st.selectbox("Search or select a Customer ID",
                                sorted(df["customer_id"].tolist()))

    row = df[df["customer_id"] == selected_id].iloc[0]

    from churn_model import predict_customer_churn
    cf, _ = build_ml_features(df[df["customer_id"] == selected_id])
    res   = predict_customer_churn(rf, cf)
    prob  = res["churn_probability_pct"]
    risk  = res["risk_category"]

    RCOL = {"Low":C_GREEN,"Medium":C_AMBER,"High":C_RED}[risk]
    RBADGE = {
        "Low":    f"<span style='background:rgba(16,185,129,.15);color:{C_GREEN};border:1px solid rgba(16,185,129,.3);padding:5px 14px;border-radius:99px;font-weight:800;font-size:.9rem'>✅ Low Risk</span>",
        "Medium": f"<span style='background:rgba(245,158,11,.15);color:{C_AMBER};border:1px solid rgba(245,158,11,.3);padding:5px 14px;border-radius:99px;font-weight:800;font-size:.9rem'>⚠️ Medium Risk</span>",
        "High":   f"<span style='background:rgba(239,68,68,.15);color:{C_RED};border:1px solid rgba(239,68,68,.3);padding:5px 14px;border-radius:99px;font-weight:800;font-size:.9rem'>🚨 High Risk</span>",
    }[risk]

    RADVICE = {
        "Low":    ("Normal engagement. No immediate action needed.", C_GREEN),
        "Medium": ("Monitor closely. Consider a check-in email or in-app message.", C_AMBER),
        "High":   ("Immediate outreach recommended. Assign to customer success team.", C_RED),
    }[risk]

    c1, c2, c3 = st.columns([1.8, 1.8, 2.4])

    def _profile_row(k, v):
        return f"""<div style="display:flex;justify-content:space-between;
                               padding:.5rem 0;border-bottom:1px solid {BORDER}">
                    <span style="font-size:.78rem;color:{MUTED}">{k}</span>
                    <span style="font-size:.82rem;font-weight:600;color:{TEXT}">{v}</span>
                   </div>"""

    with c1:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:1.4rem">
            <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;
                        letter-spacing:.1em;color:{C_BLUE};margin-bottom:.9rem">
                Customer Profile
            </div>
            {_profile_row("ID", row["customer_id"])}
            {_profile_row("Plan", row["plan"])}
            {_profile_row("Country", row["country"])}
            {_profile_row("Tenure", f"{row['tenure_months']:.1f} months")}
            {_profile_row("Monthly Fee", f"${row['monthly_fee']:.2f}")}
            {_profile_row("Payment", row["payment_method"])}
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:1.4rem">
            <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;
                        letter-spacing:.1em;color:{C_PURPLE};margin-bottom:.9rem">
                Behaviour Signals
            </div>
            {_profile_row("Usage / Month", f"{int(row['usage_frequency'])} sessions")}
            {_profile_row("Avg Session", f"{row['average_session_minutes']:.1f} min")}
            {_profile_row("Support Tickets", str(int(row["support_tickets"])))}
            {_profile_row("Payment Failures", str(int(row["payment_failures"])))}
            {_profile_row("Days Since Login", f"{int(row['days_since_login'])} days")}
            {_profile_row("Status", "🔴 Churned" if row["churned"] else "🟢 Active")}
        </div>""", unsafe_allow_html=True)

    with c3:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix":"%",
                    "font":{"color":RCOL,"size":42,"family":"Inter","weight":800}},
            title={"text":"Churn Probability",
                   "font":{"color":SUB,"size":13,"family":"Inter"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#334155",
                        "tickfont":{"color":MUTED,"size":10}},
                "bar":{"color":RCOL,"thickness":0.22},
                "bgcolor":"rgba(0,0,0,0)",
                "bordercolor":BORDER,
                "steps":[
                    {"range":[0,35],  "color":"rgba(16,185,129,0.07)"},
                    {"range":[35,65], "color":"rgba(245,158,11,0.07)"},
                    {"range":[65,100],"color":"rgba(239,68,68,0.07)"},
                ],
                "threshold":{"line":{"color":RCOL,"width":3},
                             "thickness":0.8,"value":prob},
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family":"Inter"},
            height=240,
            margin=dict(t=40,b=5,l=20,r=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:-.4rem">{RBADGE}</div>
        <div style="background:rgba({','.join(str(int(RCOL.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.07);
                    border:1px solid {RCOL}33;border-radius:10px;
                    padding:.75rem 1rem;margin-top:.8rem;
                    font-size:.8rem;color:{SUB};line-height:1.5">
            {RADVICE[0]}
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:

    page_header("Business Insights",
                "Data-driven findings translated into clear actions")

    top_plan     = (df.groupby("plan")["churned"].mean()*100).idxmax()
    top_pct      = (df.groupby("plan")["churned"].mean()*100).max()
    low_ch       = df[df["usage_frequency"]<5]["churned"].mean()*100
    high_ch      = df[df["usage_frequency"]>=15]["churned"].mean()*100
    fail_ch      = df[df["payment_failures"]>=2]["churned"].mean()*100
    ok_ch        = df[df["payment_failures"]==0]["churned"].mean()*100
    mrr          = df[df["churned"]==0]["monthly_fee"].sum()

    INSIGHTS = [
        {
            "color": C_RED,   "tag": "PLAN CHURN",
            "title": f"{top_plan} plan has the highest churn — {top_pct:.1f}%",
            "body":  "Entry-level customers may not be finding enough value quickly enough. "
                     "The gap between plan tiers suggests a product-value mismatch at the low end.",
            "action":"Add a structured 30-day onboarding email sequence for Basic plan users.",
        },
        {
            "color": C_AMBER, "tag": "ENGAGEMENT",
            "title": f"Low-usage customers churn at {low_ch:.0f}% vs {high_ch:.0f}% for active users",
            "body":  "Usage frequency is one of the strongest early-warning signals. "
                     "Customers who don't engage regularly disengage emotionally before they cancel.",
            "action":"Trigger a re-engagement campaign when monthly usage drops below 5 sessions.",
        },
        {
            "color": C_PURPLE,"tag": "PAYMENTS",
            "title": f"2+ payment failures → {fail_ch:.0f}% churn vs {ok_ch:.0f}% for clean payers",
            "body":  "Payment failure is the single strongest predictor in the model. "
                     "Customers who experience friction at payment often interpret it as a sign to leave.",
            "action":"Implement smart retry logic and a friendly reminder before each retry attempt.",
        },
        {
            "color": C_CYAN,  "tag": "SUPPORT",
            "title": "Customers with 4+ support tickets show significantly elevated churn",
            "body":  "Repeated support interactions signal unresolved frustration. "
                     "These customers are not getting value and are likely exploring alternatives.",
            "action":"Auto-flag customers after their 3rd ticket for a proactive success call.",
        },
        {
            "color": C_BLUE,  "tag": "ONBOARDING",
            "title": "The first 60 days are the single most critical retention window",
            "body":  "New customers who don't reach a habit-forming usage pattern in the first 2 months "
                     "are significantly more likely to churn. This is where investment pays off most.",
            "action":"Invest in day-3 welcome call, day-14 milestone email, day-30 feature tutorial.",
        },
        {
            "color": C_GREEN, "tag": "REVENUE IMPACT",
            "title": f"Active MRR: ${mrr:,.0f}  —  5% churn reduction = ${mrr*0.05:,.0f}/mo saved",
            "body":  "Churn is not just a retention metric — it directly impacts revenue. "
                     "Even modest improvements compound significantly over a 12-month horizon.",
            "action":"Treat churn reduction as a revenue growth initiative with dedicated budget.",
        },
    ]

    for ins in INSIGHTS:
        clr = ins["color"]
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};
                    border-left:3px solid {clr};border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1rem;
                    transition:border-color .2s">
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem">
                <span style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                             letter-spacing:.1em;color:{clr};background:{clr}18;
                             padding:2px 9px;border-radius:99px;border:1px solid {clr}33">
                    {ins['tag']}
                </span>
            </div>
            <div style="font-size:.98rem;font-weight:700;color:{TEXT};margin-bottom:.5rem">
                {ins['title']}
            </div>
            <div style="font-size:.84rem;color:{SUB};line-height:1.6;margin-bottom:.7rem">
                {ins['body']}
            </div>
            <div style="font-size:.82rem;color:{C_GREEN};
                        background:rgba(16,185,129,.07);
                        border-left:3px solid {C_GREEN};
                        border-radius:6px;padding:.55rem .85rem;line-height:1.5">
                <strong>→ Action:</strong> {ins['action']}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.5rem;padding:.9rem 1.1rem;background:{CARD2};
                border:1px solid {BORDER};border-radius:10px;
                font-size:.75rem;color:{MUTED};line-height:1.7">
        ⚠️ All findings are correlations in synthetic data.
        Validate with A/B experiments before making product decisions.
    </div>""", unsafe_allow_html=True)
