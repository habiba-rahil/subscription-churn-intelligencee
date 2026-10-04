"""
visualization.py
================
Reusable visualization functions for the project.

All functions return Plotly figures so they can be used
in both Jupyter notebooks and the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


BRAND_COLORS = {
    "primary": "#2563EB",   # blue
    "danger": "#DC2626",    # red
    "success": "#16A34A",   # green
    "warning": "#D97706",   # amber
    "neutral": "#6B7280",   # gray
}

CHURN_COLORS = ["#16A34A", "#DC2626"]  # [no churn, churn]


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing churned vs active customer counts."""
    counts = df["churned"].value_counts().reset_index()
    counts.columns = ["churned", "count"]
    counts["label"] = counts["churned"].map({0: "Active", 1: "Churned"})

    fig = px.bar(
        counts,
        x="label",
        y="count",
        color="label",
        color_discrete_sequence=CHURN_COLORS[::-1],
        title="Customer Status: Active vs Churned",
        labels={"label": "Status", "count": "Number of Customers"},
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    return fig


def plot_churn_by_plan(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing churn rate by subscription plan."""
    plan_churn = (
        df.groupby("plan")["churned"]
        .agg(["sum", "count"])
        .reset_index()
    )
    plan_churn.columns = ["plan", "churned", "total"]
    plan_churn["churn_rate"] = (plan_churn["churned"] / plan_churn["total"] * 100).round(1)

    plan_order = ["Basic", "Standard", "Premium", "Enterprise"]
    plan_churn["plan"] = pd.Categorical(plan_churn["plan"], categories=plan_order, ordered=True)
    plan_churn = plan_churn.sort_values("plan")

    fig = px.bar(
        plan_churn,
        x="plan",
        y="churn_rate",
        title="Churn Rate by Subscription Plan",
        labels={"plan": "Plan", "churn_rate": "Churn Rate (%)"},
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text="churn_rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    return fig


def plot_churn_by_country(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of churn rate by country."""
    country_churn = (
        df.groupby("country")["churned"]
        .agg(["sum", "count"])
        .reset_index()
    )
    country_churn.columns = ["country", "churned", "total"]
    country_churn["churn_rate"] = (country_churn["churned"] / country_churn["total"] * 100).round(1)
    country_churn = country_churn.sort_values("churn_rate", ascending=True)

    fig = px.bar(
        country_churn,
        y="country",
        x="churn_rate",
        orientation="h",
        title="Churn Rate by Country",
        labels={"country": "Country", "churn_rate": "Churn Rate (%)"},
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text="churn_rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    return fig


def plot_churn_by_age_group(df: pd.DataFrame) -> go.Figure:
    """Bar chart of churn rate by age group."""
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 24, 34, 44, 54, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )
    age_churn = (
        df.groupby("age_group", observed=True)["churned"]
        .agg(["sum", "count"])
        .reset_index()
    )
    age_churn.columns = ["age_group", "churned", "total"]
    age_churn["churn_rate"] = (age_churn["churned"] / age_churn["total"] * 100).round(1)

    fig = px.bar(
        age_churn,
        x="age_group",
        y="churn_rate",
        title="Churn Rate by Age Group",
        labels={"age_group": "Age Group", "churn_rate": "Churn Rate (%)"},
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text="churn_rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    return fig


def plot_churn_by_usage(df: pd.DataFrame) -> go.Figure:
    """Box plot showing usage distribution for churned vs active customers."""
    fig = px.box(
        df,
        x=df["churned"].map({0: "Active", 1: "Churned"}),
        y="usage_frequency",
        color=df["churned"].map({0: "Active", 1: "Churned"}),
        color_discrete_map={"Active": "#16A34A", "Churned": "#DC2626"},
        title="Usage Frequency: Active vs Churned Customers",
        labels={"x": "Status", "usage_frequency": "Usage Frequency (sessions/month)"},
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    return fig


def plot_churn_by_payment_method(df: pd.DataFrame) -> go.Figure:
    """Bar chart of churn rate by payment method."""
    pm_churn = (
        df.groupby("payment_method")["churned"]
        .agg(["sum", "count"])
        .reset_index()
    )
    pm_churn.columns = ["payment_method", "churned", "total"]
    pm_churn["churn_rate"] = (pm_churn["churned"] / pm_churn["total"] * 100).round(1)

    fig = px.bar(
        pm_churn,
        x="payment_method",
        y="churn_rate",
        title="Churn Rate by Payment Method",
        labels={"payment_method": "Payment Method", "churn_rate": "Churn Rate (%)"},
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text="churn_rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    return fig


def plot_cohort_heatmap(cohort_pivot: pd.DataFrame) -> go.Figure:
    """
    Create a cohort retention heatmap.

    HOW TO READ THIS CHART:
      - Each row is a cohort of customers who signed up in the same month.
      - Each column is the number of months since they signed up.
      - The colour shows what % of that cohort is still active.
      - Month 0 is always 100% (everyone just signed up).
      - As months increase, retention decreases.
    """
    # Limit to first 13 months for readability
    cols = [c for c in cohort_pivot.columns if int(c.split()[-1]) <= 12]
    data = cohort_pivot[cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=data.columns.tolist(),
            y=data.index.tolist(),
            colorscale="RdYlGn",
            text=data.values.round(1),
            texttemplate="%{text}%",
            hovertemplate="Cohort: %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            zmin=0,
            zmax=100,
        )
    )
    fig.update_layout(
        title="Cohort Retention Heatmap<br><sup>Each cell shows % of cohort still active</sup>",
        xaxis_title="Months Since Signup",
        yaxis_title="Signup Cohort",
        height=600,
    )
    return fig


def plot_segment_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing cluster distribution."""
    segment_counts = df["segment_label"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]

    fig = px.pie(
        segment_counts,
        names="segment",
        values="count",
        title="Customer Segment Distribution",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_segment_radar(profile_df: pd.DataFrame, feature_cols: list) -> go.Figure:
    """
    Radar chart comparing cluster profiles across features.

    HOW TO READ:
      Each shape represents one customer segment.
      The farther from the centre, the higher the value for that feature.
    """
    fig = go.Figure()

    for segment in profile_df.index:
        values = profile_df.loc[segment, feature_cols].tolist()
        values_norm = [(v - min(profile_df[col].min() for col in feature_cols))
                       for v, col in zip(values, feature_cols)]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_cols,
            fill="toself",
            name=str(segment),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Segment Feature Comparison",
    )
    return fig


def plot_confusion_matrix(cm: list, model_name: str = "Model") -> go.Figure:
    """Confusion matrix heatmap."""
    labels = ["No Churn (0)", "Churned (1)"]
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted: No Churn", "Predicted: Churned"],
        y=["Actual: No Churn", "Actual: Churned"],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=f"{model_name} — Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def plot_roc_curves(metrics_list: list) -> go.Figure:
    """Plot ROC curves for multiple models."""
    from sklearn.metrics import roc_curve

    fig = go.Figure()

    for m in metrics_list:
        if "y_test" in m and "y_prob" in m:
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{m['model']} (AUC={m['roc_auc']:.3f})",
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Classifier",
    ))

    fig.update_layout(
        title="ROC Curves — Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white",
    )
    return fig


def plot_churn_trend(trend_df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly churn trend."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["month"],
        y=trend_df["churned_customers"],
        mode="lines+markers",
        name="Churned Customers",
        line=dict(color="#DC2626"),
    ))

    fig.add_trace(go.Bar(
        x=trend_df["month"],
        y=trend_df["new_customers"],
        name="New Customers",
        marker_color="#2563EB",
        opacity=0.5,
        yaxis="y2",
    ))

    fig.update_layout(
        title="Monthly New Customers vs Churned Customers",
        xaxis_title="Month",
        yaxis_title="Churned Customers",
        yaxis2=dict(
            title="New Customers",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor="white",
    )
    return fig
