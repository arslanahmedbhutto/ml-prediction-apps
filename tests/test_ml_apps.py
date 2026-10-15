import json

import joblib
import pandas as pd
import pytest

from ml_apps import train
from ml_apps.tasks import DATA_DIR, MODELS_DIR, REPORTS_DIR, TASKS


def test_datasets_load_with_expected_shapes():
    assert len(TASKS["gpa"].load()) == 2392
    assert len(TASKS["house"].load()) == 489   # 11 mixed-in teaching rows excluded
    assert len(TASKS["salary"].load()) == 35


def test_house_skips_exactly_the_handmade_rows():
    raw = pd.read_csv(DATA_DIR / "house_prices.csv")
    first = raw.iloc[:11]
    assert first["Bedrooms"].iloc[5] == 5.5   # the imputed value from the hand-made file
    assert TASKS["house"].load().iloc[0].tolist() == raw.iloc[11][["Area", "Bedrooms", "Age", "Price"]].tolist()


def test_target_not_in_features():
    for t in TASKS.values():
        assert t.target not in t.features
    assert "GradeClass" not in TASKS["gpa"].features   # derived from GPA: would leak


def test_app_inputs_cover_every_feature():
    for t in TASKS.values():
        assert set(t.inputs) == set(t.features)


@pytest.mark.parametrize("key", list(TASKS))
def test_saved_models_predict_sensibly(key):
    bundle = joblib.load(MODELS_DIR / f"{key}.joblib")
    task = TASKS[key]
    X = pd.DataFrame([{f: task.inputs[f][3] for f in task.features}])[bundle["features"]]
    pred = float(bundle["model"].predict(X)[0])
    df = task.load()
    assert df[task.target].min() * 0.5 <= pred <= df[task.target].max() * 1.5


def test_directions_make_sense():
    gpa, house, salary = (joblib.load(MODELS_DIR / f"{k}.joblib") for k in ("gpa", "house", "salary"))
    base = {f: TASKS["gpa"].inputs[f][3] for f in TASKS["gpa"].features}
    more_absent = {**base, "Absences": 25}
    p = gpa["model"].predict(pd.DataFrame([base, more_absent])[gpa["features"]])
    assert p[1] < p[0]
    h = house["model"].predict(pd.DataFrame([{"Area": 2500, "Bedrooms": 5, "Age": 10},
                                             {"Area": 3500, "Bedrooms": 5, "Age": 10}]))
    assert h[1] > h[0]
    s = salary["model"].predict(pd.DataFrame({"YearsExperience": [2.0, 10.0]}))
    assert s[1] > s[0]


def test_every_model_beats_the_mean_baseline():
    tasks = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))["tasks"]
    for s in tasks.values():
        base = next(r for r in s["results"] if r["model"] == "Mean baseline")["cv_rmse"]
        assert all(r["cv_rmse"] < base for r in s["results"] if r["model"] != "Mean baseline")


def test_candidates_cover_linear_and_tree_models():
    names = set(train.candidates(100))
    assert {"Mean baseline", "Linear regression", "Random forest", "Gradient boosting"} <= names
