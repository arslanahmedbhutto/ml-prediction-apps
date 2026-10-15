"""Compare regression models for each task, save the best one, write reports and figures.

    python -m ml_apps.train
"""

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_apps.tasks import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, TASKS, Task

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True,
})


def candidates(n_rows: int) -> dict:
    small = n_rows < 100
    return {
        "Mean baseline": DummyRegressor(),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3 if small else 2, random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200 if small else 400, max_depth=2 if small else 3,
                                                       learning_rate=0.05, random_state=SEED),
    }


def evaluate_task(task: Task) -> dict:
    df = task.load()
    X, y = df[task.features], df[task.target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, model in candidates(len(df)).items():
        cvr = cross_validate(model, X_tr, y_tr, cv=cv, scoring=("neg_root_mean_squared_error", "r2"))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({
            "model": name,
            "cv_rmse": float(-cvr["test_neg_root_mean_squared_error"].mean()),
            "cv_rmse_std": float(cvr["test_neg_root_mean_squared_error"].std()),
            "cv_r2": float(cvr["test_r2"].mean()),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "test_mae": float(mean_absolute_error(y_te, pred)),
            "test_r2": float(r2_score(y_te, pred)),
        })
        fitted[name] = model

    rows.sort(key=lambda r: r["cv_rmse"])
    best = next(r for r in rows if r["model"] != "Mean baseline")["model"]
    model = fitted[best]
    pred = model.predict(X_te)
    resid_sd = float(np.std(y_te - pred, ddof=1))

    imp = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=SEED, scoring="r2")
    importance = pd.Series(imp.importances_mean, index=task.features).sort_values(ascending=False)

    extra = {}
    if task.key == "gpa":
        one = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr[["Absences"]], y_tr)
        extra["absences_only_test_r2"] = float(r2_score(y_te, one.predict(X_te[["Absences"]])))
        extra["absences_corr"] = float(df["Absences"].corr(df["GPA"]))
    if hasattr(model, "named_steps") and "linearregression" in model.named_steps:
        lr, sc = model.named_steps["linearregression"], model.named_steps["standardscaler"]
        extra["coefficients_per_unit"] = dict(zip(task.features, map(float, lr.coef_ / sc.scale_)))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": task.features, "model_name": best, "test_rmse": rows[[r["model"] for r in rows].index(best)]["test_rmse"],
                 "residual_sd": resid_sd, "train_ranges": {f: [float(X[f].min()), float(X[f].max())] for f in task.features}},
                MODELS_DIR / f"{task.key}.joblib", compress=3)

    figures(task, rows, best, y_te, pred, importance)
    return {"task": task.title, "rows": len(df), "best_model": best, "results": rows,
            "permutation_importance_r2_drop": importance.round(4).to_dict(), "residual_sd": resid_sd, **extra}


def figures(task, rows, best, y_te, pred, importance):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    names = [r["model"] for r in rows][::-1]
    vals = [r["cv_rmse"] for r in rows][::-1]
    errs = [r["cv_rmse_std"] for r in rows][::-1]
    axes[0].barh(names, vals, xerr=errs, color=[BLUE if n == best else MUTED for n in names], height=0.6,
                 error_kw={"ecolor": INK_2, "capsize": 3})
    for i, v in enumerate(vals):
        axes[0].text(v + errs[i] + max(vals) * 0.02, i, f"{v:,.3g}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.25)
    axes[0].set_title("Cross-validated RMSE (lower is better)")
    axes[0].grid(axis="y", visible=False)

    axes[1].scatter(y_te, pred, s=14, color=BLUE, alpha=0.6, edgecolor="none")
    lo, hi = float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "--", color=ORANGE, lw=1.5)
    axes[1].set_xlabel(f"Actual {task.target_label}")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"{best}: test set (R² = {r2_score(y_te, pred):.3f})")

    imp = importance.head(8)[::-1]
    axes[2].barh(imp.index, imp.values, color=BLUE, height=0.6)
    axes[2].set_title("Permutation importance (drop in R²)")
    axes[2].grid(axis="y", visible=False)

    fig.suptitle(task.title, x=0.01, ha="left", fontweight="bold", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{task.key}_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def markdown(summary: dict) -> str:
    out = ["# Model comparison\n", "5-fold CV on the training split (80%); test = held-out 20%. Best model chosen by CV RMSE.\n"]
    for key, s in summary.items():
        out.append(f"\n## {s['task']} ({s['rows']:,} rows), best: **{s['best_model']}**\n")
        out.append("| Model | CV RMSE | CV R² | Test RMSE | Test MAE | Test R² |\n|---|---|---|---|---|---|")
        for r in s["results"]:
            name = f"**{r['model']}**" if r["model"] == s["best_model"] else r["model"]
            out.append(f"| {name} | {r['cv_rmse']:,.3f} ± {r['cv_rmse_std']:,.3f} | {r['cv_r2']:.3f} | "
                       f"{r['test_rmse']:,.3f} | {r['test_mae']:,.3f} | {r['test_r2']:.3f} |")
    return "\n".join(out) + "\n"


def main():
    summary = {}
    for key, task in TASKS.items():
        s = evaluate_task(task)
        summary[key] = s
        print(f"{task.title:<14} best {s['best_model']:<18} test R² "
              f"{next(r for r in s['results'] if r['model'] == s['best_model'])['test_r2']:.3f}")
    REPORTS_DIR.mkdir(exist_ok=True)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({**meta, "tasks": summary}, indent=2), encoding="utf-8")
    md = markdown(summary)
    (REPORTS_DIR / "model_comparison.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
