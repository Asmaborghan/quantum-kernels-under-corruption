"""Interactive explorer for "Bandwidth-Tuned Quantum Kernels under Data Corruption".

Run locally:  streamlit run app/streamlit_app.py
Everything live is computed with the project's own engines (quantum_kernel.py, corruption.py);
everything else is read from the saved results in results/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantum_kernel import load_ready, get_states, kernel_from_states  # noqa: E402
from corruption import corrupt_training, noise_columns, LEVELS          # noqa: E402

RES = ROOT / "results"
REPO = "https://github.com/Asmaborghan/quantum-kernels-under-corruption"
KAGGLE = "https://www.kaggle.com/code/asmaborghan/quantum-kernels-under-data-corruption"

DATASETS = {
    "German Credit": "german_credit", "Concrete": "concrete", "Water Potability": "water_potability",
    "Steel Plates Faults": "steel_faults", "Phoneme": "phoneme", "MAGIC Telescope": "magic_telescope",
    "Bank Marketing": "bank_marketing", "Electricity": "electricity",
}
SHORT = {"german_credit": "German", "concrete": "Concrete", "water_potability": "Water", "steel_faults": "Steel",
         "phoneme": "Phoneme", "magic_telescope": "MAGIC", "bank_marketing": "Bank", "electricity": "Electricity"}
MAGMA = ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"]
C_GRID = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
DAMAGE_LABEL = {"missing": "Missing values (MCAR + median fill)", "noise": "Gaussian noise (× feature SD)",
                "labels": "Wrong labels (random flips)"}
MODEL_LABEL = {"quantum (c,C tuned)": "Quantum (c, C tuned)", "svm_rbf": "SVM-RBF",
               "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
COLORS = {"Quantum (c, C tuned)": "#6929C4", "SVM-RBF": "#E8590C", "Random Forest": "#2F9E44",
          "Logistic Regression": "#1971C2"}

st.set_page_config(page_title="Quantum Kernels under Data Corruption", page_icon="⚛️", layout="wide")


# ---------------------------------------------------------------- cached helpers
@st.cache_data
def data(name):
    X, y, _, _ = load_ready(name)
    order = np.argsort(y, kind="stable")          # class 0 rows first: block structure is visible
    return X[order], y[order].astype(int)


@st.cache_data
def kernel(name, n_rows, c, kind, level, seed):
    X, y = data(name)
    idx = np.linspace(0, len(X) - 1, n_rows).astype(int)
    Xs, ys = X[idx], y[idx]
    if kind != "clean" and level > 0:
        Xs, ys = corrupt_training(Xs, ys, kind, level, seed, noise_columns(name))
    S = get_states(Xs, Xs.shape[1], c)
    return kernel_from_states(S, S), ys


def off(K):
    return K[np.triu_indices_from(K, k=1)]


@st.cache_data
def folds(name):
    return pd.read_csv(RES / "runs" / f"{name}_folds.csv")


@st.cache_data
def table(file, **kw):
    return pd.read_csv(RES / "analysis" / file, **kw)


# ---------------------------------------------------------------- header + sidebar
st.title("⚛️ Quantum Kernels under Data Corruption")
st.caption("Asma Mohamed · Rayyan Aljabri · Thoria Al-aisri, UTAS Sohar. "
           f"Interactive companion to the paper. [Code]({REPO}) · [Beginner notebook]({KAGGLE})")

with st.sidebar:
    st.header("Settings")
    ds_label = st.selectbox("Dataset", list(DATASETS), index=0)
    ds = DATASETS[ds_label]
    X_all, y_all = data(ds)
    st.caption(f"{X_all.shape[1]} features → **{X_all.shape[1]} qubits** · 280 training rows")
    kind = st.radio("Damage type", ["clean", "missing", "noise", "labels"],
                    format_func=lambda k: "Clean" if k == "clean" else DAMAGE_LABEL[k])
    level = 0.0
    if kind != "clean":
        level = st.select_slider("Damage level", options=list(LEVELS[kind]), value=LEVELS[kind][-1],
                                 format_func=lambda v: f"{v:g} × SD" if kind == "noise" else f"{v:.0%}")
    c = st.select_slider("Bandwidth c", options=C_GRID, value=0.1)
    n_rows = st.slider("Rows shown in the kernel", 40, 200, 80, step=20)
    seed = st.number_input("Damage seed", 0, 99, 0)
    st.divider()
    st.caption("ZZ feature map · linear entanglement · 2 repetitions · fidelity kernel "
               "K(x, x′) = |⟨φ(x)|φ(x′)⟩|², exact statevector simulation.")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Kernel lab", "📉 Bandwidth & concentration",
                                  "📊 Results explorer", "🎯 Shot noise & hardware"])

# ---------------------------------------------------------------- 1. kernel lab
with tab1:
    K0, y0 = kernel(ds, n_rows, c, "clean", 0.0, seed)
    K1, y1 = kernel(ds, n_rows, c, kind, level, seed)
    v0, v1 = off(K0).var(), off(K1).var()
    m = st.columns(4)
    m[0].metric("Mean similarity (clean)", f"{off(K0).mean():.3f}")
    m[1].metric("Variance (clean)", f"{v0:.2e}")
    m[2].metric("Variance (damaged)", f"{v1:.2e}")
    m[3].metric("Variance ratio damaged / clean", f"{v1 / v0:.2f}",
                help="< 1: damage flattens the kernel (more concentration). > 1: the kernel gets spikier.")

    left, right = st.columns(2)
    for col, K, title in [(left, K0, "Clean"), (right, K1, "Damaged" if kind != "clean" else "Clean")]:
        fig_m, ax_m = plt.subplots(figsize=(5, 4.4), dpi=110)
        im = ax_m.imshow(K, cmap="magma", vmin=0, vmax=1)
        ax_m.set_title(f"{title} kernel · c = {c}", fontsize=11); ax_m.set_xticks([]); ax_m.set_yticks([])
        fig_m.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04, label="K(x, x′)")
        col.pyplot(fig_m, clear_figure=True)
        plt.close(fig_m)

    if kind == "labels":
        flipped = int((y1 != y0).sum())
        st.info(f"Wrong labels do not change the kernel at all: the features are untouched. "
                f"{flipped} of {n_rows} labels shown here are flipped, so the damage only reaches the SVM "
                "through the targets it is trained on.")
    hist = go.Figure()
    hist.add_histogram(x=off(K0), name="clean", opacity=0.6, nbinsx=50)
    if kind != "clean":
        hist.add_histogram(x=off(K1), name="damaged", opacity=0.6, nbinsx=50)
    hist.update_layout(barmode="overlay", height=300, title="Off-diagonal similarities",
                       xaxis_title="K(x, x′)", yaxis_title="pairs", margin=dict(t=40, b=10))
    st.plotly_chart(hist, width="stretch", theme=None, key="chart2")
    st.caption("Rows are sorted by class (class 0 first), so a useful kernel shows two brighter blocks. "
               "Damage is applied with the project's own `corrupt_training`; noise only touches measured columns.")

# ---------------------------------------------------------------- 2. bandwidth & concentration
with tab2:
    st.subheader("How the kernel flattens as c grows")
    rows = []
    for cc in C_GRID:
        Ka, _ = kernel(ds, n_rows, cc, "clean", 0.0, seed)
        rows.append(("clean", cc, off(Ka).var()))
        if kind not in ("clean", "labels"):
            Kb, _ = kernel(ds, n_rows, cc, kind, level, seed)
            rows.append(("damaged", cc, off(Kb).var()))
    sweep = pd.DataFrame(rows, columns=["data", "c", "variance"])
    fig = px.line(sweep, x="c", y="variance", color="data", markers=True, log_x=True, log_y=True)
    fig.add_vline(x=c, line_dash="dot", line_color="gray")
    fig.update_layout(height=380, yaxis_title="off-diagonal variance (log)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch", theme=None, key="chart3")
    st.caption("Small variance means a flat kernel. At large c it is flat because every pair looks different "
               "(concentration); at very small c it is flat because every pair looks the same.")

    st.subheader("From the study: variance ratio under the strongest damage (all 280 rows)")
    vr = table("e2_variance_ratio.csv")
    cols = [k for k in vr.columns if k.startswith("c =")]
    long = vr.melt(id_vars=["dataset", "setting"], value_vars=cols, var_name="c", value_name="ratio")
    long["c"] = long["c"].str.replace("c = ", "").astype(float)
    pick = st.radio("Damage", sorted(long.setting.unique()), horizontal=True)
    fig = px.line(long[long.setting == pick], x="c", y="ratio", color="dataset", markers=True, log_x=True, log_y=True)
    fig.add_hline(y=1, line_color="black", line_width=1)
    fig.update_layout(height=400, yaxis_title="variance ratio damaged / clean", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch", theme=None, key="chart4")
    st.caption("Noise flattens the kernel at large c (Concrete ×0.05, Steel ×0.04 at c = 1); missing values with "
               "median fill make it spikier (Bank ×5.7 at c = 0.5). At the small c that tuning picks, both effects are small.")

# ---------------------------------------------------------------- 3. results explorer
with tab3:
    f = folds(ds)
    st.subheader(f"{ds_label}: ROC-AUC against bandwidth c (25 outer folds)")
    sel_kind, sel_level = (kind, level) if kind != "clean" else ("clean", 0.0)
    sub = f[(f.kind == sel_kind) & np.isclose(f.level, sel_level)]
    qc = sub[sub.model == "quantum (c fixed)"].groupby("c").auc.agg(["mean", "sem"]).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=qc["c"], y=qc["mean"], error_y=dict(array=1.96 * qc["sem"]), mode="lines+markers",
                    name="Quantum at fixed c", line=dict(color="#6929C4"))
    for mdl, lab in MODEL_LABEL.items():
        val = sub[sub.model == mdl].auc.mean()
        fig.add_scatter(x=[C_GRID[0], C_GRID[-1]], y=[val, val], mode="lines", name=f"{lab}: {val:.3f}",
                        line=dict(dash="dash", color=COLORS[lab]))
    fig.update_xaxes(type="log", title="bandwidth c"); fig.update_yaxes(title="ROC-AUC")
    fig.update_layout(height=430, margin=dict(t=20), legend=dict(orientation="h", y=-0.3), margin_b=80)
    st.plotly_chart(fig, width="stretch", theme=None, key="chart5")
    st.caption(f"Setting: {'clean' if sel_kind == 'clean' else f'{sel_kind} {sel_level:g}'} (sidebar). "
               "Error bars: 95% interval of the fold mean (not corrected for overlapping folds).")

    st.subheader("AUC change from clean data as damage grows")
    scope = st.radio("Scope", [ds_label, "Mean of all 8 datasets"], horizontal=True)
    names = [ds] if scope == ds_label else list(DATASETS.values())
    curves = []
    for nm in names:
        d = folds(nm)
        d = d[d.model.isin(MODEL_LABEL)]
        base = d[d.kind == "clean"].groupby("model").auc.mean()
        for dk in ["missing", "noise", "labels"]:
            g = d[d.kind == dk].groupby(["model", "level"]).auc.mean().reset_index()
            g["change"] = g.auc - g.model.map(base)
            g["damage"], g["dataset"] = dk, nm
            curves.append(pd.concat([pd.DataFrame({"model": base.index, "level": 0.0, "auc": base.values,
                                                   "change": 0.0, "damage": dk, "dataset": nm}), g]))
    cv = pd.concat(curves).groupby(["damage", "model", "level"]).change.mean().reset_index()
    cv["model"] = cv.model.map(MODEL_LABEL)
    fig = px.line(cv, x="level", y="change", color="model", facet_col="damage", markers=True,
                  color_discrete_map=COLORS, category_orders={"damage": ["missing", "noise", "labels"]})
    fig.update_xaxes(matches=None, title="level"); fig.update_yaxes(title="AUC change")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=420, margin=dict(t=40, b=90), legend=dict(orientation="h", y=-0.3, title=None))
    st.plotly_chart(fig, width="stretch", theme=None, key="chart6")
    st.caption("Pooled over the eight datasets, wrong labels cost about 0.03 AUC for both quantum and SVM-RBF at 15%, "
               "and the quantum − RBF trend is ≈ 0 for every damage type.")

    with st.expander("Study tables"):
        st.markdown("**Clean data (mean ± SD, 25 folds)**")
        st.dataframe(table("table_clean.csv", index_col=0), width="stretch")
        st.markdown("**Pooled quantum − SVM-RBF per setting (random effects, Hartung–Knapp)**")
        st.dataframe(table("stats_pooled.csv"), width="stretch")
        st.markdown("**Best fixed bandwidth per setting**")
        st.dataframe(table("table_rq1_best_c.csv", index_col=0), width="stretch")

# ---------------------------------------------------------------- 4. shot noise & hardware
with tab4:
    st.subheader("What a real device would measure")
    shots = st.select_slider("Shots per similarity", options=[50, 100, 300, 1000, 3000, 10000], value=1000)
    Kc, _ = kernel(ds, min(n_rows, 60), c, "clean", 0.0, seed)
    exact = off(Kc)
    est = np.random.default_rng(seed).binomial(shots, np.clip(exact, 0, 1)) / shots
    m = st.columns(3)
    m[0].metric("Mean |error|", f"{np.abs(est - exact).mean():.4f}")
    m[1].metric("Rank agreement (Spearman)", f"{pd.Series(exact).corr(pd.Series(est), method='spearman'):.3f}")
    m[2].metric("Pairs estimated as exactly 0", f"{(est == 0).mean():.0%}")
    fig = px.scatter(x=exact, y=est, opacity=0.5, labels={"x": "exact K", "y": f"estimate from {shots} shots"})
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), showlegend=False)
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch", theme=None, key="chart7")
    st.caption("Each similarity is estimated as Binomial(shots, K) / shots: the share of all-zero outcomes of the "
               "compute–uncompute circuit. Small similarities, common at large c, often round to exactly 0.")

    st.subheader("From the study")
    a, b = st.columns(2)
    with a:
        st.markdown("**Shot noise, pooled over 8 datasets**")
        st.dataframe(table("shots_pooled.csv"), width="stretch", height=330)
    with b:
        st.markdown("**IBM `ibm_fez` vs the simulated device (FakeFez), 4,000 shots**")
        st.dataframe(table("ibm_hardware_summary.csv", index_col=0).round(3), width="stretch")
        img = RES / "analysis" / "fig_ibm_hardware_all.png"
        if img.exists():
            st.image(str(img), width="stretch")
    st.caption("Shot noise costs 0.01–0.05 AUC. Real hardware keeps the order of similarities "
               "(rank agreement 0.88–0.99 at c = 0.1) but shrinks them: self-similarity ≈ 0.7 (5–8 qubits), ≈ 0.35 (14 qubits).")

st.divider()
st.caption(f"Code, data and all fold scores: [{REPO.replace('https://', '')}]({REPO}). MIT licence.")
