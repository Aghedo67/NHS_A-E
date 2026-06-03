import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NHS A&E Performance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #003087; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #768692; margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #f0f4f8; border-radius: 10px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .metric-box h3 { margin: 0; font-size: 1.8rem; color: #003087; }
    .metric-box p  { margin: 0; font-size: 0.8rem; color: #768692; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #003087; border-left: 4px solid #005EB8;
        padding-left: 0.6rem; margin: 1.5rem 0 0.8rem;
    }
    .badge-green  { background:#d4edda; color:#155724; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-amber  { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-red    { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .stDataFrame { font-size: 0.82rem; }
    div[data-testid="stSidebarContent"] { background: #003087; }
    div[data-testid="stSidebarContent"] * { color: white !important; }
    div[data-testid="stSidebarContent"] .stSelectbox label,
    div[data-testid="stSidebarContent"] .stSlider label { color: #b0c4de !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("Provider_Level_Data.csv")
    df = df[df["Code"] != "-"].copy().reset_index(drop=True)

    col_map = {
        "A&E attendances - Type 1 Departments - Major A&E":        "type1_attendances",
        "Type 2 Departments - Single Specialty":                    "type2_attendances",
        "Type 3 Departments - Other A&E/Minor Injury Unit":         "type3_attendances",
        "Total attendances":                                        "total_attendances",
        "Total Attendances < 4 hours":                              "total_lt4h",
        "Total Attendances > 4 hours":                              "total_gt4h",
        "Percentage of attendances within 4 hours - Percentage in 4 hours or less (all)": "pct_within_4h_all",
        "Percentage in 4 hours or less (type 1)":                   "pct_within_4h_t1",
        "Percentage in 4 hours or less (type 2)":                   "pct_within_4h_t2",
        "Percentage in 4 hours or less (type 3)":                   "pct_within_4h_t3",
        "Emergency Admissions - Emergency Admissions via Type 1 A&E":"emerg_admissions_t1",
        "Total Emergency Admissions via A&E":                       "total_emerg_admissions_ae",
        "Other Emergency admissions (i.e not via A&E)":             "other_emerg_admissions",
        "Total Emergency Admissions":                               "total_emerg_admissions",
        "Number of patients spending >4 hours from decision to admit to admission":  "pts_gt4h_dta",
        "Number of patients spending >12 hours from decision to admit to admission": "pts_gt12h_dta",
    }
    df.rename(columns=col_map, inplace=True)

    for c in ["pct_within_4h_all","pct_within_4h_t1","pct_within_4h_t2","pct_within_4h_t3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    s = lambda a, b: a / b.replace(0, np.nan)
    df["admission_rate"] = s(df["total_emerg_admissions"], df["total_attendances"])
    df["t1_share"]       = s(df["type1_attendances"],      df["total_attendances"])
    df["t2_share"]       = s(df["type2_attendances"],      df["total_attendances"])
    df["t3_share"]       = s(df["type3_attendances"],      df["total_attendances"])
    df["dta_4h_rate"]    = s(df["pts_gt4h_dta"],           df["total_attendances"])
    df["dta_12h_rate"]   = s(df["pts_gt12h_dta"],          df["total_attendances"])
    df["gt4h_rate"]      = s(df["total_gt4h"],             df["total_attendances"])

    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["Region"].fillna("Unknown"))

    median_perf = df["pct_within_4h_all"].median()
    df["target"] = (df["pct_within_4h_all"] >= median_perf).astype(int)

    return df, le, median_perf

@st.cache_resource
def train_models(df):
    FEATURES = [
        "total_attendances","type1_attendances","type2_attendances","type3_attendances",
        "admission_rate","t1_share","t2_share","t3_share",
        "dta_4h_rate","dta_12h_rate","total_emerg_admissions","other_emerg_admissions","region_enc",
    ]
    df_m = df.dropna(subset=FEATURES + ["target"]).copy()
    X, y = df_m[FEATURES], df_m["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    X_s    = scaler.transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_tr, y_tr);   gb.fit(X_tr, y_tr);   lr.fit(X_tr_s, y_tr)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    results = {
        "Random Forest":       { "model": rf,  "X_te": X_te,   "y_te": y_te,
            "pred": rf.predict(X_te),  "prob": rf.predict_proba(X_te)[:,1],
            "cv_auc": cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean(),
            "fi": pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False) },
        "Gradient Boosting":   { "model": gb,  "X_te": X_te,   "y_te": y_te,
            "pred": gb.predict(X_te),  "prob": gb.predict_proba(X_te)[:,1],
            "cv_auc": cross_val_score(gb, X, y, cv=cv, scoring="roc_auc").mean() },
        "Logistic Regression": { "model": lr,  "X_te": X_te_s, "y_te": y_te,
            "pred": lr.predict(X_te_s),"prob": lr.predict_proba(X_te_s)[:,1],
            "cv_auc": cross_val_score(lr, X_s, y, cv=cv, scoring="roc_auc").mean() },
    }

    # Full predictions using LR (best CV AUC)
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    lr_full  = LogisticRegression(max_iter=1000, random_state=42)
    lr_full.fit(X_full_s, y)
    df_m = df_m.copy()
    df_m["pred_class"] = lr_full.predict(X_full_s)
    df_m["pred_prob"]  = lr_full.predict_proba(X_full_s)[:,1]

    return results, df_m, FEATURES, scaler, scaler_full, lr_full

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    df, le, median_perf = load_and_prepare()
    results, df_pred, FEATURES, scaler, scaler_full, lr_full = train_models(df)

regions   = sorted(df["Region"].dropna().unique())
providers = sorted(df["Name"].dropna().unique())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 NHS A&E Predictor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 EDA",
        "🤖 Model Results",
        "🔮 Predict Provider",
        "📋 Provider Explorer",
    ])
    st.markdown("---")
    st.markdown("**Dataset:** Q4 2025/26")
    st.markdown(f"**Providers:** {len(df)}")
    st.markdown(f"**Median 4-hr perf:** {median_perf:.1%}")
    st.markdown("---")
    st.markdown("**Filter (EDA & Explorer)**")
    sel_region = st.multiselect("Region", regions, default=regions)

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
NHS_BLUE   = "#005EB8"
NHS_GREEN  = "#009639"
NHS_AMBER  = "#FFB81C"
NHS_RED    = "#DA291C"
NHS_DARK   = "#003087"

def perf_color(p):
    if p >= 0.9:  return NHS_GREEN
    if p >= 0.76: return NHS_AMBER
    return NHS_RED

def perf_badge(p):
    pct = p * 100
    if p >= 0.9:  return f'<span class="badge-green">✓ {pct:.1f}%</span>'
    if p >= 0.76: return f'<span class="badge-amber">⚠ {pct:.1f}%</span>'
    return f'<span class="badge-red">✗ {pct:.1f}%</span>'

filt_df = df[df["Region"].isin(sel_region)].copy()

# ═══════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="main-title">NHS A&E Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quarter 4 2025/26 · England · Provider-level analysis</div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total providers", f"{len(df):,}", "NHS trusts & centres"),
        ("Total attendances", f"{df['total_attendances'].sum()/1e6:.2f}M", "Q4 2025/26"),
        ("Avg 4-hr performance", f"{df['pct_within_4h_all'].mean():.1%}", "National average"),
        ("Meeting interim target", f"{(df['pct_within_4h_all']>=0.76).mean():.1%}", "≥76% threshold"),
        ("Median wait target", f"{median_perf:.1%}", "Classifier threshold"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f'<div class="metric-box"><h3>{val}</h3><p><b>{label}</b></p><p>{sub}</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # Regional performance + attendance pie
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Regional 4-hr performance</div>', unsafe_allow_html=True)
        reg = (df.groupby("Region")["pct_within_4h_all"].mean()
               .sort_values().reset_index())
        reg["region_short"] = reg["Region"].str.replace("NHS England ","")
        reg["color"] = reg["pct_within_4h_all"].apply(perf_color)
        fig = px.bar(reg, x="pct_within_4h_all", y="region_short",
                     orientation="h", color="pct_within_4h_all",
                     color_continuous_scale=["#DA291C","#FFB81C","#009639"],
                     range_color=[0.7, 0.9],
                     labels={"pct_within_4h_all":"Avg 4-hr performance","region_short":""},
                     text=reg["pct_within_4h_all"].apply(lambda x: f"{x:.1%}"))
        fig.add_vline(x=0.76, line_dash="dot", line_color="grey", annotation_text="76% target")
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=300, margin=dict(l=0,r=20,t=10,b=0),
                          xaxis=dict(tickformat=".0%", range=[0.68,0.95]))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Attendance share by region</div>', unsafe_allow_html=True)
        att = df.groupby("Region")["total_attendances"].sum().reset_index()
        att["region_short"] = att["Region"].str.replace("NHS England ","")
        fig2 = px.pie(att, values="total_attendances", names="region_short",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Bottom performers table
    st.markdown('<div class="section-header">10 providers with lowest 4-hr performance</div>', unsafe_allow_html=True)
    bot = (df.nsmallest(10, "pct_within_4h_all")
           [["Name","Region","total_attendances","pct_within_4h_all","admission_rate","dta_12h_rate"]]
           .copy())
    bot.columns = ["Provider","Region","Total attendances","4-hr %","Admission rate","12-hr DTA rate"]
    bot["4-hr %"]        = (bot["4-hr %"]*100).round(1).astype(str) + "%"
    bot["Admission rate"] = (bot["Admission rate"]*100).round(1).astype(str) + "%"
    bot["12-hr DTA rate"] = (bot["12-hr DTA rate"]*100).round(2).astype(str) + "%"
    bot["Total attendances"] = bot["Total attendances"].apply(lambda x: f"{x:,}")
    st.dataframe(bot, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Showing {len(filt_df)} providers from selected regions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🗺️ Scatter Analysis", "🔥 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">4-hr performance distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(filt_df.dropna(subset=["pct_within_4h_all"]),
                               x="pct_within_4h_all", nbins=25,
                               color_discrete_sequence=[NHS_BLUE],
                               labels={"pct_within_4h_all":"% within 4 hours"})
            fig.add_vline(x=median_perf, line_dash="dash", line_color=NHS_RED,
                          annotation_text=f"Median {median_perf:.1%}")
            fig.add_vline(x=0.76, line_dash="dot", line_color=NHS_AMBER,
                          annotation_text="Target 76%")
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0),
                              xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Total attendances distribution</div>', unsafe_allow_html=True)
            fig2 = px.histogram(filt_df, x="total_attendances", nbins=30,
                                color_discrete_sequence=[NHS_DARK],
                                labels={"total_attendances":"Total attendances"})
            fig2.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Performance by department type mix</div>', unsafe_allow_html=True)
        filt_df2 = filt_df.copy()
        filt_df2["dept_type"] = np.where(filt_df2["t1_share"]>0.5,"Mainly Type 1 (Major A&E)",
                                np.where(filt_df2["t3_share"]>0.5,"Mainly Type 3 (Minor/UTC)","Mixed"))
        fig3 = px.box(filt_df2.dropna(subset=["pct_within_4h_all"]),
                      x="dept_type", y="pct_within_4h_all",
                      color="dept_type",
                      color_discrete_map={
                          "Mainly Type 1 (Major A&E)": NHS_RED,
                          "Mixed": NHS_AMBER,
                          "Mainly Type 3 (Minor/UTC)": NHS_GREEN,
                      },
                      labels={"pct_within_4h_all":"% within 4 hours","dept_type":""})
        fig3.add_hline(y=0.76, line_dash="dot", line_color="grey")
        fig3.update_layout(height=350, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                           yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Attendances vs 4-hr performance</div>', unsafe_allow_html=True)
        sc_df = filt_df.dropna(subset=["pct_within_4h_all","admission_rate"]).copy()
        sc_df["admission_rate_pct"] = sc_df["admission_rate"] * 100
        sc_df["perf_pct"] = sc_df["pct_within_4h_all"] * 100
        fig = px.scatter(sc_df,
                         x="total_attendances", y="perf_pct",
                         color="admission_rate_pct",
                         size="total_attendances",
                         hover_name="Name",
                         hover_data={"total_attendances":True,"perf_pct":":.1f",
                                     "admission_rate_pct":":.1f","Region":True},
                         color_continuous_scale="RdYlGn_r",
                         labels={"total_attendances":"Total attendances",
                                 "perf_pct":"4-hr performance (%)",
                                 "admission_rate_pct":"Admission rate (%)"},
                         size_max=30)
        fig.add_hline(y=median_perf*100, line_dash="dash", line_color=NHS_RED, opacity=0.6)
        fig.add_hline(y=76, line_dash="dot", line_color=NHS_AMBER, opacity=0.6)
        fig.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Admission rate vs 4-hr performance</div>', unsafe_allow_html=True)
            fig4 = px.scatter(sc_df, x="admission_rate_pct", y="perf_pct",
                              color="Region", hover_name="Name",
                              trendline="ols",
                              labels={"admission_rate_pct":"Admission rate (%)","perf_pct":"4-hr %"})
            fig4.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig4, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">12-hr DTA rate vs 4-hr performance</div>', unsafe_allow_html=True)
            sc_df["dta_12h_pct"] = sc_df["dta_12h_rate"] * 100
            fig5 = px.scatter(sc_df, x="dta_12h_pct", y="perf_pct",
                              color="Region", hover_name="Name",
                              trendline="ols",
                              labels={"dta_12h_pct":"12-hr DTA rate (%)","perf_pct":"4-hr %"})
            fig5.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Feature correlation matrix</div>', unsafe_allow_html=True)
        corr_cols = ["pct_within_4h_all","t1_share","t3_share","admission_rate",
                     "dta_4h_rate","dta_12h_rate","gt4h_rate","total_attendances"]
        corr_labels = ["4-hr perf","T1 share","T3 share","Adm rate",
                       "DTA 4h","DTA 12h",">4h rate","Volume"]
        corr_df = filt_df[corr_cols].dropna().copy()
        corr_df.columns = corr_labels
        corr_matrix = corr_df.corr()

        fig_c = px.imshow(corr_matrix,
                          color_continuous_scale="RdYlGn",
                          zmin=-1, zmax=1,
                          text_auto=".2f",
                          aspect="auto")
        fig_c.update_traces(textfont_size=11)
        fig_c.update_layout(height=440, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("**Key takeaways from the correlation matrix:**")
        st.markdown("- Type 1 department share has a **strong negative** correlation with 4-hr performance — major A&E depts struggle most")
        st.markdown("- Decision-to-admit (DTA) rates are **strongly negative** — bed flow is a critical bottleneck")
        st.markdown("- Type 3 share is **positively** correlated — minor injury units achieve near-perfect 4-hr rates")

# ═══════════════════════════════════════════
# PAGE 3 – MODEL RESULTS
# ═══════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<div class="main-title">Predictive Model Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Binary classification: above vs below national median ({median_perf:.1%}) 4-hr performance</div>', unsafe_allow_html=True)

    # Model comparison cards
    st.markdown('<div class="section-header">Model comparison</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, res) in zip(cols, results.items()):
        acc = accuracy_score(res["y_te"], res["pred"])
        auc = roc_auc_score(res["y_te"], res["prob"])
        best = name == "Logistic Regression"
        border = f"border: 2px solid {NHS_BLUE};" if best else "border: 1px solid #dee2e6;"
        col.markdown(f"""
        <div style="background:white; {border} border-radius:10px; padding:1rem; text-align:center">
            {"⭐ " if best else ""}<b>{name}</b>{"  (best)" if best else ""}
            <hr style="margin:0.5rem 0">
            <div style="display:flex; justify-content:space-around; margin-top:0.5rem">
                <div><div style="font-size:1.4rem;font-weight:700;color:{NHS_BLUE}">{acc:.1%}</div><div style="font-size:0.75rem;color:#768692">Accuracy</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:{NHS_DARK}">{auc:.3f}</div><div style="font-size:0.75rem;color:#768692">ROC-AUC</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:{NHS_GREEN}">{res['cv_auc']:.3f}</div><div style="font-size:0.75rem;color:#768692">CV AUC</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["📊 ROC Curves", "🌲 Feature Importances", "🔢 Confusion Matrix"])

    with tab1:
        fig = go.Figure()
        colors_roc = [NHS_BLUE, NHS_GREEN, NHS_AMBER]
        for (name, res), color in zip(results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(res["y_te"], res["prob"])
            auc = roc_auc_score(res["y_te"], res["prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                                     line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random baseline",
                                  line=dict(color="grey", dash="dash", width=1)))
        fig.update_layout(height=420, xaxis_title="False positive rate",
                          yaxis_title="True positive rate",
                          legend=dict(x=0.6, y=0.1),
                          margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rf_res = results["Random Forest"]
        fi = rf_res["fi"].sort_values(ascending=True)
        fi_labels = {
            "total_attendances":"Total attendances","type1_attendances":"Type 1 attendances",
            "type2_attendances":"Type 2 attendances","type3_attendances":"Type 3 attendances",
            "admission_rate":"Admission rate","t1_share":"Type 1 dept share",
            "t2_share":"Type 2 dept share","t3_share":"Type 3 dept share",
            "dta_4h_rate":"4-hr DTA rate","dta_12h_rate":"12-hr DTA rate",
            "total_emerg_admissions":"Total emerg. admissions",
            "other_emerg_admissions":"Other emerg. admissions","region_enc":"Region",
        }
        fi.index = [fi_labels.get(i,i) for i in fi.index]
        fig2 = px.bar(x=fi.values * 100, y=fi.index, orientation="h",
                      color=fi.values * 100,
                      color_continuous_scale=["#cce5ff","#005EB8","#003087"],
                      labels={"x":"Importance (%)","y":""},
                      text=[f"{v:.1f}%" for v in fi.values*100])
        fig2.update_coloraxes(showscale=False)
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=420, margin=dict(l=0,r=40,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

        st.info("💡 **Type 1 department share** is the most important predictor — major A&E departments handle the most complex cases and face the greatest flow pressures.")

    with tab3:
        sel_model = st.selectbox("Select model", list(results.keys()))
        res = results[sel_model]
        cm = confusion_matrix(res["y_te"], res["pred"])
        cm_df = pd.DataFrame(cm,
                             index=["Actual: Below median","Actual: Above median"],
                             columns=["Pred: Below median","Pred: Above median"])
        fig3 = px.imshow(cm_df, text_auto=True,
                         color_continuous_scale=[[0,"white"],[1,NHS_BLUE]],
                         aspect="auto")
        fig3.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3, use_container_width=True)

        report = classification_report(res["y_te"], res["pred"],
                                       target_names=["Below median","Above median"],
                                       output_dict=True)
        rep_df = pd.DataFrame(report).T.round(3)
        st.dataframe(rep_df, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 4 – PREDICT PROVIDER
# ═══════════════════════════════════════════
elif page == "🔮 Predict Provider":
    st.markdown('<div class="main-title">Provider Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter provider metrics to predict whether it will perform above or below the national median.</div>', unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Select existing provider", "Enter custom values"], horizontal=True)

    if mode == "Select existing provider":
        selected = st.selectbox("Choose a provider", providers)
        prov_data = df[df["Name"] == selected].iloc[0]

        t1 = int(prov_data["type1_attendances"])
        t2 = int(prov_data["type2_attendances"])
        t3 = int(prov_data["type3_attendances"])
        total = int(prov_data["total_attendances"])
        emerg = int(prov_data["total_emerg_admissions"])
        other_emerg = int(prov_data["other_emerg_admissions"])
        dta4  = float(prov_data["dta_4h_rate"]) if not pd.isna(prov_data["dta_4h_rate"]) else 0.0
        dta12 = float(prov_data["dta_12h_rate"]) if not pd.isna(prov_data["dta_12h_rate"]) else 0.0
        region_val = prov_data["Region"]
        actual_perf = prov_data["pct_within_4h_all"]

        st.info(f"**Actual 4-hr performance:** {actual_perf:.1%}" if not pd.isna(actual_perf) else "Actual performance: N/A")

    else:
        c1, c2 = st.columns(2)
        with c1:
            t1    = st.number_input("Type 1 attendances (Major A&E)", 0, 200000, 30000, step=1000)
            t2    = st.number_input("Type 2 attendances (Single Specialty)", 0, 50000, 0, step=500)
            t3    = st.number_input("Type 3 attendances (Minor Injury/UTC)", 0, 200000, 20000, step=1000)
            total = t1 + t2 + t3
            st.metric("Total attendances", f"{total:,}")
        with c2:
            emerg       = st.number_input("Total emergency admissions", 0, 100000, 10000, step=500)
            other_emerg = st.number_input("Other emergency admissions (not via A&E)", 0, 50000, 5000, step=500)
            dta4        = st.slider("4-hr decision-to-admit rate", 0.0, 0.3, 0.05, 0.005, format="%.3f")
            dta12       = st.slider("12-hr decision-to-admit rate", 0.0, 0.15, 0.02, 0.001, format="%.3f")
        region_val = st.selectbox("Region", regions)
        actual_perf = None

    # Build feature vector
    if total > 0:
        t1_sh = t1 / total
        t2_sh = t2 / total
        t3_sh = t3 / total
        adm_r = emerg / total
    else:
        t1_sh = t2_sh = t3_sh = adm_r = 0.0

    region_enc_val = le.transform([region_val])[0] if region_val in le.classes_ else 0

    X_input = pd.DataFrame([{
        "total_attendances":      total,
        "type1_attendances":      t1,
        "type2_attendances":      t2,
        "type3_attendances":      t3,
        "admission_rate":         adm_r,
        "t1_share":               t1_sh,
        "t2_share":               t2_sh,
        "t3_share":               t3_sh,
        "dta_4h_rate":            dta4,
        "dta_12h_rate":           dta12,
        "total_emerg_admissions": emerg,
        "other_emerg_admissions": other_emerg,
        "region_enc":             region_enc_val,
    }])

    if st.button("🔮 Predict Performance", type="primary"):
        X_sc = scaler_full.transform(X_input)
        pred_class = lr_full.predict(X_sc)[0]
        pred_prob  = lr_full.predict_proba(X_sc)[0][1]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", "Above median ✅" if pred_class == 1 else "Below median ⚠️")
        r2.metric("Confidence (above median)", f"{pred_prob:.1%}")
        r3.metric("National median baseline", f"{median_perf:.1%}")

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            title={"text": "Probability of above-median performance"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": NHS_GREEN if pred_class==1 else NHS_RED},
                "steps": [
                    {"range": [0, 50],  "color": "#fde8e8"},
                    {"range": [50, 75], "color": "#fff3cd"},
                    {"range": [75, 100],"color": "#d4edda"},
                ],
                "threshold": {"line": {"color": NHS_DARK, "width": 3}, "value": 50}
            },
            number={"suffix": "%", "font": {"size": 36}},
        ))
        fig_g.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        # Feature breakdown
        st.markdown('<div class="section-header">Input feature summary</div>', unsafe_allow_html=True)
        feature_summary = pd.DataFrame({
            "Feature": ["Type 1 share","Type 3 share","Admission rate","4-hr DTA rate","12-hr DTA rate","Total attendances"],
            "Your value": [f"{t1_sh:.1%}", f"{t3_sh:.1%}", f"{adm_r:.1%}", f"{dta4:.2%}", f"{dta12:.2%}", f"{total:,}"],
            "National median": [
                f"{df['t1_share'].median():.1%}",
                f"{df['t3_share'].median():.1%}",
                f"{df['admission_rate'].median():.1%}",
                f"{df['dta_4h_rate'].median():.2%}",
                f"{df['dta_12h_rate'].median():.2%}",
                f"{int(df['total_attendances'].median()):,}",
            ]
        })
        st.dataframe(feature_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# PAGE 5 – PROVIDER EXPLORER
# ═══════════════════════════════════════════
elif page == "📋 Provider Explorer":
    st.markdown('<div class="main-title">Provider Explorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Browse and filter all {len(filt_df)} providers · ML predictions included</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    perf_min = c1.slider("Min 4-hr performance (%)", 0, 100, 0)
    perf_max = c2.slider("Max 4-hr performance (%)", 0, 100, 100)
    sort_by   = c3.selectbox("Sort by", ["4-hr performance ↓","4-hr performance ↑","Total attendances ↓","Name"])

    sort_map = {
        "4-hr performance ↓": ("pct_within_4h_all", False),
        "4-hr performance ↑": ("pct_within_4h_all", True),
        "Total attendances ↓": ("total_attendances", False),
        "Name": ("Name", True),
    }
    sort_col, sort_asc = sort_map[sort_by]

    exp_df = (filt_df
              .dropna(subset=["pct_within_4h_all"])
              .copy())
    exp_df = exp_df[
        (exp_df["pct_within_4h_all"] * 100 >= perf_min) &
        (exp_df["pct_within_4h_all"] * 100 <= perf_max)
    ].sort_values(sort_col, ascending=sort_asc)

    # Merge predictions
    if "pred_prob" in df_pred.columns:
        exp_df = exp_df.merge(
            df_pred[["Code","pred_class","pred_prob"]],
            on="Code", how="left"
        )

    display_cols = {
        "Name": "Provider",
        "Region": "Region",
        "total_attendances": "Attendances",
        "pct_within_4h_all": "4-hr perf",
        "admission_rate": "Admission rate",
        "dta_12h_rate": "12-hr DTA",
        "pred_prob": "ML score",
    }
    show = exp_df[[c for c in display_cols if c in exp_df.columns]].copy()
    show.rename(columns=display_cols, inplace=True)
    if "4-hr perf" in show.columns:
        show["4-hr perf"]       = (show["4-hr perf"]*100).round(1).astype(str) + "%"
    if "Admission rate" in show.columns:
        show["Admission rate"]  = (show["Admission rate"]*100).round(1).astype(str) + "%"
    if "12-hr DTA" in show.columns:
        show["12-hr DTA"]       = (show["12-hr DTA"]*100).round(2).astype(str) + "%"
    if "ML score" in show.columns:
        show["ML score"]        = (show["ML score"]*100).round(0).astype(int).astype(str) + "%"
    if "Attendances" in show.columns:
        show["Attendances"]     = show["Attendances"].apply(lambda x: f"{int(x):,}")

    st.markdown(f"**{len(show)} providers match your filters**")
    st.dataframe(show, use_container_width=True, hide_index=True, height=480)

    # Download button
    csv_out = exp_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_out,
        file_name="nhs_ae_filtered_providers.csv",
        mime="text/csv",
    )
