import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NHS A&E Regression Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title { font-size:2rem; font-weight:700; color:#003087; margin-bottom:0.2rem; }
.sub-title  { font-size:1rem; color:#768692; margin-bottom:1.5rem; }
.kpi-box    { background:#f0f4f8; border-radius:10px; padding:1rem 1.2rem; text-align:center; }
.kpi-box h3 { margin:0; font-size:1.8rem; color:#003087; }
.kpi-box p  { margin:0; font-size:0.8rem; color:#768692; }
.section-hdr{ font-size:1.05rem; font-weight:600; color:#003087;
              border-left:4px solid #005EB8; padding-left:0.6rem; margin:1.5rem 0 0.8rem; }
</style>
""", unsafe_allow_html=True)

NHS_BLUE = "#005EB8"; NHS_GREEN = "#009639"; NHS_RED = "#DA291C"; NHS_AMB = "#FFB81C"

# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Provider_Level_Data.csv")
    df = df[df["Code"] != "-"].copy().reset_index(drop=True)
    col_map = {
        "A&E attendances - Type 1 Departments - Major A&E":       "type1_attendances",
        "Type 2 Departments - Single Specialty":                   "type2_attendances",
        "Type 3 Departments - Other A&E/Minor Injury Unit":        "type3_attendances",
        "Total attendances":                                       "total_attendances",
        "Total Attendances > 4 hours":                             "total_gt4h",
        "Percentage of attendances within 4 hours - Percentage in 4 hours or less (all)": "pct_within_4h_all",
        "Emergency Admissions - Emergency Admissions via Type 1 A&E": "emerg_admissions_t1",
        "Other Emergency admissions (i.e not via A&E)":            "other_emerg_admissions",
        "Total Emergency Admissions":                              "total_emerg_admissions",
        "Number of patients spending >4 hours from decision to admit to admission":  "pts_gt4h_dta",
        "Number of patients spending >12 hours from decision to admit to admission": "pts_gt12h_dta",
    }
    df.rename(columns=col_map, inplace=True)
    df["pct_within_4h_all"] = pd.to_numeric(df["pct_within_4h_all"], errors="coerce")
    s = lambda a, b: a / b.replace(0, np.nan)
    df["admission_rate"] = s(df["total_emerg_admissions"], df["total_attendances"])
    df["t1_share"]       = s(df["type1_attendances"],      df["total_attendances"])
    df["t2_share"]       = s(df["type2_attendances"],      df["total_attendances"])
    df["t3_share"]       = s(df["type3_attendances"],      df["total_attendances"])
    df["dta_4h_rate"]    = s(df["pts_gt4h_dta"],           df["total_attendances"])
    df["dta_12h_rate"]   = s(df["pts_gt12h_dta"],          df["total_attendances"])
    df["log_total"]      = np.log1p(df["total_attendances"])
    df["t1_x_adm"]       = df["t1_share"] * df["admission_rate"]
    df["dta_x_t1"]       = df["dta_4h_rate"] * df["t1_share"]
    df["high_complexity"]= ((df["t1_share"]>0.5)&(df["admission_rate"]>0.2)).astype(int)
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["Region"].fillna("Unknown"))
    return df, le

@st.cache_resource
def train_all(df):
    FEATURES = ["total_attendances","log_total","type1_attendances","type2_attendances",
                "type3_attendances","admission_rate","t1_share","t2_share","t3_share",
                "dta_4h_rate","dta_12h_rate","total_emerg_admissions","other_emerg_admissions",
                "region_enc","t1_x_adm","dta_x_t1","high_complexity"]
    TARGET = "pct_within_4h_all"
    df_m = df.dropna(subset=FEATURES+[TARGET]).copy().reset_index(drop=True)
    X, y = df_m[FEATURES], df_m[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te); X_s = scaler.transform(X)
    kf = KFold(5, shuffle=True, random_state=42)

    models_def = {
        "Ridge":             (Ridge(alpha=10.0), True),
        "ElasticNet":        (ElasticNet(alpha=0.01, l1_ratio=0.5), True),
        "Random Forest":     (RandomForestRegressor(n_estimators=300, max_depth=6,
                                                    min_samples_leaf=3, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                        learning_rate=0.03, subsample=0.8,
                                                        random_state=42), False),
        "XGBoost":           (XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           subsample=0.8, random_state=42, verbosity=0), False),
    }
    results = {}
    for name, (mdl, is_lin) in models_def.items():
        Xtr, Xte, Xall = (X_tr_s, X_te_s, X_s) if is_lin else (X_tr, X_te, X)
        mdl.fit(Xtr, y_tr)
        pred = np.clip(mdl.predict(Xte), 0, 1)
        results[name] = {
            "model": mdl, "is_linear": is_lin,
            "rmse": np.sqrt(mean_squared_error(y_te, pred))*100,
            "mae":  mean_absolute_error(y_te, pred)*100,
            "r2":   r2_score(y_te, pred),
            "cv_r2":cross_val_score(mdl, Xall, y, cv=kf, scoring="r2").mean(),
            "pred_test": pred, "y_test": y_te,
        }
    best = max(results, key=lambda k: results[k]["cv_r2"])
    best_mdl = results[best]["model"]
    Xfull = X_s if results[best]["is_linear"] else X
    df_m = df_m.copy()
    df_m["predicted_4h_pct"] = np.clip(best_mdl.predict(Xfull), 0, 1)
    df_m["residual"]         = df_m[TARGET] - df_m["predicted_4h_pct"]
    df_m["abs_error_pct"]    = df_m["residual"].abs() * 100
    df_m["performance_gap"]  = df_m["predicted_4h_pct"] - 0.76
    return results, df_m, FEATURES, scaler, best, best_mdl

with st.spinner("Loading data and training models…"):
    df, le = load_data()
    results, df_pred, FEATURES, scaler, best_name, best_mdl = train_all(df)

regions   = sorted(df["Region"].dropna().unique())
providers = sorted(df["Name"].dropna().unique())

FI_LABELS = {
    "dta_4h_rate":"4-hr DTA rate","dta_x_t1":"DTA × T1 share",
    "type1_attendances":"Type 1 attendances","total_emerg_admissions":"Total emerg. admissions",
    "t1_x_adm":"T1 share × Admission rate","t1_share":"Type 1 dept share",
    "dta_12h_rate":"12-hr DTA rate","admission_rate":"Admission rate",
    "t3_share":"Type 3 dept share","log_total":"Log total attendances",
    "total_attendances":"Total attendances","type2_attendances":"Type 2 attendances",
    "type3_attendances":"Type 3 attendances","t2_share":"Type 2 dept share",
    "other_emerg_admissions":"Other emerg. admissions","region_enc":"Region",
    "high_complexity":"High complexity flag",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 A&E Regression")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🤖 Model Results",
        "🔍 Residual Analysis",
        "🔮 Predict Provider",
        "📋 Provider Explorer",
    ])
    st.markdown("---")
    st.markdown(f"**Best model:** {best_name}")
    st.markdown(f"**CV R²:** {results[best_name]['cv_r2']:.4f}")
    st.markdown(f"**Test RMSE:** {results[best_name]['rmse']:.2f}%")
    st.markdown(f"**Test MAE:** {results[best_name]['mae']:.2f}%")
    st.markdown("---")
    sel_regions = st.multiselect("Filter regions", regions, default=regions)

filt = df_pred[df_pred["Region"].isin(sel_regions)].copy()

# ═══════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="main-title">NHS A&E Regression Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predicting exact 4-hr performance % per provider · Q4 2025/26</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, label, sub) in zip([c1,c2,c3,c4,c5], [
        (f"{len(df_pred):,}",         "Providers modelled",     "after cleaning"),
        (f"{results[best_name]['cv_r2']:.4f}", f"CV R² ({best_name})", "5-fold cross-val"),
        (f"{results[best_name]['rmse']:.2f}%", "Test RMSE",      "avg prediction error"),
        (f"{results[best_name]['mae']:.2f}%",  "Test MAE",       "mean abs error"),
        (f"{(filt['abs_error_pct']<2).mean():.0%}", "Within 2% error", "of predictions"),
    ]):
        col.markdown(f'<div class="kpi-box"><h3>{val}</h3><p><b>{label}</b></p><p>{sub}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-hdr">Actual vs predicted performance</div>', unsafe_allow_html=True)
        fig = px.scatter(filt, x="pct_within_4h_all", y="predicted_4h_pct",
                         color="Region", hover_name="Name",
                         hover_data={"abs_error_pct":":.2f"},
                         labels={"pct_within_4h_all":"Actual 4-hr %",
                                 "predicted_4h_pct":"Predicted 4-hr %"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.add_trace(go.Scatter(x=[0.45,1.05], y=[0.45,1.05], mode="lines",
                                 line=dict(dash="dash", color="grey", width=1.5),
                                 showlegend=False))
        fig.update_layout(height=370, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Prediction error distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(filt, x="abs_error_pct", nbins=30,
                            color_discrete_sequence=[NHS_BLUE],
                            labels={"abs_error_pct":"Absolute error (%)"})
        fig2.add_vline(x=filt["abs_error_pct"].mean(), line_dash="dash",
                       line_color=NHS_RED,
                       annotation_text=f"Mean {filt['abs_error_pct'].mean():.2f}%")
        fig2.update_layout(height=370, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-hdr">Performance gap vs 76% target (predicted)</div>', unsafe_allow_html=True)
    gap_df = filt.copy()
    gap_df["gap_pct"] = gap_df["performance_gap"] * 100
    gap_df["colour"]  = gap_df["gap_pct"].apply(lambda x: NHS_GREEN if x >= 0 else NHS_RED)
    gap_df_sorted = gap_df.sort_values("gap_pct").head(40)
    fig3 = px.bar(gap_df_sorted, x="Name", y="gap_pct",
                  color="gap_pct",
                  color_continuous_scale=["#DA291C","#FFB81C","#009639"],
                  labels={"gap_pct":"Gap vs 76% target (pp)","Name":""},
                  hover_data={"Region":True,"pct_within_4h_all":":.1%","predicted_4h_pct":":.1%"})
    fig3.add_hline(y=0, line_color="black", line_width=1)
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(height=340, xaxis_tickangle=-45, margin=dict(l=0,r=0,t=10,b=120))
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 2 – MODEL RESULTS
# ═══════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<div class="main-title">Model Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">5 regression models compared on test set and 5-fold cross-validation</div>', unsafe_allow_html=True)

    # Model scorecards
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        is_best = name == best_name
        border = f"2px solid {NHS_BLUE}" if is_best else "1px solid #dee2e6"
        col.markdown(f"""
        <div style="background:white;border:{border};border-radius:10px;padding:0.8rem;text-align:center">
            {"⭐ " if is_best else ""}<b style="font-size:0.85rem">{name}</b>
            <hr style="margin:0.4rem 0">
            <div style="font-size:1.2rem;font-weight:700;color:{NHS_BLUE}">{res['r2']:.4f}</div>
            <div style="font-size:0.7rem;color:#768692">Test R²</div>
            <div style="margin-top:6px;font-size:1rem;font-weight:600;color:#003087">{res['cv_r2']:.4f}</div>
            <div style="font-size:0.7rem;color:#768692">CV R²</div>
            <div style="margin-top:6px;font-size:0.9rem;color:{NHS_RED}">{res['rmse']:.2f}%</div>
            <div style="font-size:0.7rem;color:#768692">RMSE</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2 = st.tabs(["📊 Feature importances", "📈 Model comparison"])

    with tab1:
        rf_res = results["Random Forest"]
        fi = pd.Series(rf_res["model"].feature_importances_, index=FEATURES)
        fi.index = [FI_LABELS.get(i, i) for i in fi.index]
        fi = fi.sort_values(ascending=True)
        fig = px.bar(x=fi.values*100, y=fi.index, orientation="h",
                     color=fi.values*100,
                     color_continuous_scale=["#cce5ff","#005EB8","#003087"],
                     labels={"x":"Importance (%)","y":""},
                     text=[f"{v:.1f}%" for v in fi.values*100])
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=500, margin=dict(l=0,r=60,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **4-hr DTA rate** and its interaction with Type 1 share together explain ~45% of the model's predictive power — bed flow is the dominant lever.")

    with tab2:
        model_names = list(results.keys())
        cv_vals   = [results[n]["cv_r2"]  for n in model_names]
        test_vals = [results[n]["r2"]     for n in model_names]
        rmse_vals = [results[n]["rmse"]   for n in model_names]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="CV R²",   x=model_names, y=cv_vals,   marker_color=NHS_BLUE))
        fig2.add_trace(go.Bar(name="Test R²", x=model_names, y=test_vals, marker_color=NHS_GREEN))
        fig2.update_layout(barmode="group", height=360, yaxis=dict(range=[0.85,0.96]),
                           margin=dict(l=0,r=0,t=20,b=0), legend=dict(x=0.7,y=0.1))
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(x=model_names, y=rmse_vals, color=rmse_vals,
                      color_continuous_scale=["#009639","#FFB81C","#DA291C"],
                      labels={"x":"Model","y":"RMSE (%)"},
                      text=[f"{v:.2f}%" for v in rmse_vals])
        fig3.update_coloraxes(showscale=False)
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=300, title="RMSE by model (lower = better)",
                           margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 3 – RESIDUAL ANALYSIS
# ═══════════════════════════════════════════
elif page == "🔍 Residual Analysis":
    st.markdown('<div class="main-title">Residual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Diagnosing model errors to understand where predictions break down</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-hdr">Residuals vs predicted</div>', unsafe_allow_html=True)
        fig = px.scatter(filt, x="predicted_4h_pct", y="residual",
                         color="Region", hover_name="Name",
                         labels={"predicted_4h_pct":"Predicted 4-hr %","residual":"Residual"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Error band breakdown</div>', unsafe_allow_html=True)
        bands  = ["<2%","2-5%","5-10%",">10%"]
        counts = [
            (filt["abs_error_pct"] < 2).sum(),
            ((filt["abs_error_pct"]>=2)&(filt["abs_error_pct"]<5)).sum(),
            ((filt["abs_error_pct"]>=5)&(filt["abs_error_pct"]<10)).sum(),
            (filt["abs_error_pct"]>=10).sum(),
        ]
        fig2 = px.pie(values=counts, names=bands,
                      color_discrete_sequence=[NHS_GREEN, NHS_BLUE, NHS_AMB, NHS_RED])
        fig2.update_traces(textinfo="percent+label", pull=[0.05,0,0,0.1])
        fig2.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-hdr">Largest prediction errors — providers to investigate</div>', unsafe_allow_html=True)
    worst = filt.nlargest(15, "abs_error_pct")[
        ["Name","Region","pct_within_4h_all","predicted_4h_pct","residual","abs_error_pct",
         "t1_share","admission_rate","dta_4h_rate"]
    ].copy()
    worst["pct_within_4h_all"]  = (worst["pct_within_4h_all"]*100).round(1).astype(str) + "%"
    worst["predicted_4h_pct"]   = (worst["predicted_4h_pct"]*100).round(1).astype(str) + "%"
    worst["residual"]            = (worst["residual"]*100).round(2).astype(str) + "%"
    worst["abs_error_pct"]       = worst["abs_error_pct"].round(2).astype(str) + "%"
    worst["t1_share"]            = (worst["t1_share"]*100).round(1).astype(str) + "%"
    worst["admission_rate"]      = (worst["admission_rate"]*100).round(1).astype(str) + "%"
    worst["dta_4h_rate"]         = (worst["dta_4h_rate"]*100).round(2).astype(str) + "%"
    worst.columns = ["Provider","Region","Actual","Predicted","Residual","Abs Error",
                     "T1 Share","Adm Rate","DTA 4h"]
    st.dataframe(worst, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# PAGE 4 – PREDICT PROVIDER
# ═══════════════════════════════════════════
elif page == "🔮 Predict Provider":
    st.markdown('<div class="main-title">Live Provider Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter operational metrics to predict exact 4-hr performance percentage</div>', unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Select existing provider", "Enter custom values"], horizontal=True)

    if mode == "Select existing provider":
        selected = st.selectbox("Choose a provider", providers)
        row = df[df["Name"] == selected].iloc[0]
        t1   = int(row["type1_attendances"])
        t2   = int(row["type2_attendances"])
        t3   = int(row["type3_attendances"])
        total= int(row["total_attendances"])
        emerg= int(row["total_emerg_admissions"])
        other= int(row["other_emerg_admissions"])
        dta4 = float(row["dta_4h_rate"]) if not pd.isna(row.get("dta_4h_rate",np.nan)) else 0.0
        dta12= float(row["dta_12h_rate"]) if not pd.isna(row.get("dta_12h_rate",np.nan)) else 0.0
        reg  = row["Region"]
        actual = row["pct_within_4h_all"]
        if not pd.isna(actual):
            st.info(f"**Actual recorded 4-hr performance:** {actual:.1%}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            t1    = st.number_input("Type 1 attendances",   0, 200000, 30000, 1000)
            t2    = st.number_input("Type 2 attendances",   0,  50000,     0,  500)
            t3    = st.number_input("Type 3 attendances",   0, 200000, 20000, 1000)
            total = t1 + t2 + t3
            st.metric("Total attendances", f"{total:,}")
        with c2:
            emerg = st.number_input("Total emergency admissions",     0, 100000, 10000, 500)
            other = st.number_input("Other emergency admissions",     0,  50000,  5000, 500)
            dta4  = st.slider("4-hr decision-to-admit rate",  0.0, 0.30, 0.05, 0.005, format="%.3f")
            dta12 = st.slider("12-hr decision-to-admit rate", 0.0, 0.15, 0.02, 0.001, format="%.3f")
        reg    = st.selectbox("Region", regions)
        actual = None

    if st.button("📈 Predict 4-hr Performance", type="primary"):
        safe = lambda a, b: a / b if b > 0 else 0.0
        t1s  = safe(t1, total); t2s = safe(t2, total); t3s = safe(t3, total)
        admr = safe(emerg, total)
        log_t = np.log1p(total)
        reg_enc = le.transform([reg])[0] if reg in le.classes_ else 0
        hc   = int(t1s > 0.5 and admr > 0.2)

        X_in = pd.DataFrame([{
            "total_attendances": total, "log_total": log_t,
            "type1_attendances": t1, "type2_attendances": t2, "type3_attendances": t3,
            "admission_rate": admr, "t1_share": t1s, "t2_share": t2s, "t3_share": t3s,
            "dta_4h_rate": dta4, "dta_12h_rate": dta12,
            "total_emerg_admissions": emerg, "other_emerg_admissions": other,
            "region_enc": reg_enc, "t1_x_adm": t1s*admr, "dta_x_t1": dta4*t1s,
            "high_complexity": hc,
        }])

        pred_val = float(np.clip(best_mdl.predict(X_in), 0, 1)[0])

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted 4-hr performance", f"{pred_val:.1%}")
        r2.metric("vs 76% interim target", f"{(pred_val-0.76)*100:+.1f}pp",
                  delta_color="normal" if pred_val >= 0.76 else "inverse")
        r3.metric("vs national median (77.6%)", f"{(pred_val-0.776)*100:+.1f}pp",
                  delta_color="normal" if pred_val >= 0.776 else "inverse")
        if actual and not pd.isna(actual):
            r4.metric("Actual (recorded)", f"{actual:.1%}",
                      f"Model error: {(pred_val-actual)*100:+.1f}pp")

        # Gauge
        colour = NHS_GREEN if pred_val >= 0.76 else (NHS_AMB if pred_val >= 0.65 else NHS_RED)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_val * 100,
            delta={"reference": 76, "suffix": "pp vs 76%", "valueformat": "+.1f"},
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Predicted 4-hr performance"},
            gauge={
                "axis": {"range": [40, 100], "ticksuffix": "%"},
                "bar": {"color": colour},
                "steps": [
                    {"range": [40, 65],  "color": "#fde8e8"},
                    {"range": [65, 76],  "color": "#fff3cd"},
                    {"range": [76, 100], "color": "#d4edda"},
                ],
                "threshold": {"line": {"color": "#003087","width": 3}, "value": 76},
            },
        ))
        fig_g.update_layout(height=320, margin=dict(l=20,r=20,t=60,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        # What-if sliders
        st.markdown('<div class="section-hdr">What-if simulator — adjust key levers</div>', unsafe_allow_html=True)
        st.caption("Drag the sliders to see how operational changes would shift predicted performance.")
        w1, w2, w3 = st.columns(3)
        new_dta4  = w1.slider("Change 4-hr DTA rate",  0.0, 0.3,  float(dta4),  0.005, format="%.3f")
        new_admr  = w2.slider("Change admission rate",  0.0, 0.6,  float(admr),  0.01,  format="%.2f")
        new_t1s   = w3.slider("Change Type 1 share",    0.0, 1.0,  float(t1s),   0.01,  format="%.2f")

        X_wi = X_in.copy()
        X_wi["dta_4h_rate"]  = new_dta4
        X_wi["admission_rate"] = new_admr
        X_wi["t1_share"]     = new_t1s
        X_wi["t1_x_adm"]     = new_t1s * new_admr
        X_wi["dta_x_t1"]     = new_dta4 * new_t1s
        X_wi["high_complexity"] = int(new_t1s > 0.5 and new_admr > 0.2)
        wi_pred = float(np.clip(best_mdl.predict(X_wi), 0, 1)[0])
        delta_pp = (wi_pred - pred_val) * 100

        st.metric("Revised prediction", f"{wi_pred:.1%}",
                  f"{delta_pp:+.2f}pp vs original",
                  delta_color="normal" if delta_pp >= 0 else "inverse")

# ═══════════════════════════════════════════
# PAGE 5 – PROVIDER EXPLORER
# ═══════════════════════════════════════════
elif page == "📋 Provider Explorer":
    st.markdown('<div class="main-title">Provider Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Browse all providers with actual and predicted performance</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    p_min = c1.slider("Min actual 4-hr %", 0, 100, 0)
    p_max = c2.slider("Max actual 4-hr %", 0, 100, 100)
    sort_by = c3.selectbox("Sort by", [
        "Actual 4-hr % ↓","Actual 4-hr % ↑","Abs error ↓","Predicted ↓","Name"
    ])

    sort_map = {
        "Actual 4-hr % ↓":  ("pct_within_4h_all",  False),
        "Actual 4-hr % ↑":  ("pct_within_4h_all",  True),
        "Abs error ↓":       ("abs_error_pct",       False),
        "Predicted ↓":       ("predicted_4h_pct",    False),
        "Name":              ("Name",                True),
    }
    sc, sa = sort_map[sort_by]

    show = (filt.dropna(subset=["pct_within_4h_all"])
            .copy())
    show = show[
        (show["pct_within_4h_all"]*100 >= p_min) &
        (show["pct_within_4h_all"]*100 <= p_max)
    ].sort_values(sc, ascending=sa)

    display = show[["Name","Region","total_attendances","pct_within_4h_all",
                     "predicted_4h_pct","abs_error_pct","performance_gap",
                     "t1_share","admission_rate","dta_4h_rate"]].copy()

    display.columns = ["Provider","Region","Attendances","Actual 4-hr %",
                       "Predicted 4-hr %","Abs Error","Gap vs 76%",
                       "T1 Share","Adm Rate","DTA 4h"]
    for c in ["Actual 4-hr %","Predicted 4-hr %","T1 Share","Adm Rate","DTA 4h"]:
        display[c] = (display[c]*100).round(1).astype(str) + "%"
    display["Abs Error"]   = display["Abs Error"].round(2).astype(str) + "%"
    display["Gap vs 76%"]  = (show["performance_gap"]*100).round(1).astype(str) + "pp"
    display["Attendances"] = show["total_attendances"].apply(lambda x: f"{int(x):,}")

    st.markdown(f"**{len(display)} providers match your filters**")
    st.dataframe(display, use_container_width=True, hide_index=True, height=500)

    csv = show.to_csv(index=False)
    st.download_button("⬇️ Download filtered data as CSV", csv,
                       "nhs_ae_regression_filtered.csv", "text/csv")
