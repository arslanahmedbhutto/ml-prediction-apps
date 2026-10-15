"""Three regression models in one Streamlit app.

    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ml_apps.tasks import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TASKS

st.set_page_config(page_title="ML Prediction Apps", page_icon="📈", layout="wide")


@st.cache_resource
def load(key):
    return joblib.load(MODELS_DIR / f"{key}.joblib")


@st.cache_data
def metrics():
    return json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))["tasks"]


def fmt(task, v):
    return f"{v:.2f}" if task.key == "gpa" else f"{v:,.0f}"


def number_input(task, feat, col):
    label, lo, hi, default, step = task.inputs[feat]
    if isinstance(step, int) and hi - lo <= 1:
        return col.selectbox(label, [0, 1], index=default, key=f"{task.key}-{feat}")
    return col.number_input(label, min_value=lo, max_value=hi, value=default, step=step, key=f"{task.key}-{feat}")


st.title("📈 ML Prediction Apps")
st.write("Three regression models, each picked by cross-validation from five candidates. "
         "Predictions show a typical error range from the held-out test set.")

info = metrics()
tabs = st.tabs([f"{'🎓' if k == 'gpa' else '🏠' if k == 'house' else '💼'} {t.title}" for k, t in TASKS.items()])

for tab, (key, task) in zip(tabs, TASKS.items()):
    with tab:
        bundle, m = load(key), info[key]
        best = next(r for r in m["results"] if r["model"] == m["best_model"])
        st.caption(f"Model: **{m['best_model']}** · test R² **{best['test_r2']:.3f}** · "
                   f"typical error ±{fmt(task, best['test_mae'])} · trained on {m['rows']:,} rows")

        left, right = st.columns([1, 1.3])
        with left:
            cols = st.columns(2 if len(task.features) > 3 else 1)
            values = {f: number_input(task, f, cols[i % len(cols)]) for i, f in enumerate(task.features)}

        X = pd.DataFrame([values])[task.features]
        pred = float(bundle["model"].predict(X)[0])
        band = 1.96 * bundle["residual_sd"]
        if key == "gpa":
            pred = float(np.clip(pred, 0, 4))

        with right:
            st.metric(f"Predicted {task.target_label}", fmt(task, pred))
            st.caption(f"95% of test-set errors were within about ±{fmt(task, band)}.")
            outside = [f for f, (lo, hi) in bundle["train_ranges"].items() if not lo <= values[f] <= hi]
            if outside:
                st.warning(f"Outside the training data for: {', '.join(outside)}. The model is extrapolating.")

            # What-if: vary the most important feature, keep the others fixed
            top = max(m["permutation_importance_r2_drop"], key=m["permutation_importance_r2_drop"].get)
            lo, hi = bundle["train_ranges"][top]
            grid = pd.DataFrame([values] * 40)[task.features]
            grid[top] = np.linspace(lo, hi, 40)
            curve = pd.DataFrame({top: grid[top], f"Predicted {task.target_label}": bundle["model"].predict(grid)})
            st.write(f"**What if {task.inputs[top][0].lower()} changed?** (other inputs fixed)")
            st.line_chart(curve, x=top, height=230)

        with st.expander("How the model was chosen"):
            st.dataframe(pd.DataFrame(m["results"]).round(3), hide_index=True)
            img = FIGURES_DIR / f"{key}_results.png"
            if img.exists():
                st.image(str(img))

st.caption("Educational project. GPA and house-price data are synthetic; see the README for data notes.")
