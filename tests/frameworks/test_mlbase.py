import subprocess
import sys

import pandas as pd
import pytest
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlrun import new_function
from mlrun.frameworks.sklearn import apply_mlrun


def _is_installed(lib) -> bool:
    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    return lib not in installed_packages


def get_dataset(classification=True):
    if classification:
        iris = load_iris()
        X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        y = pd.DataFrame(data=iris.target, columns=["species"])
    else:
        boston = load_boston()
        X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        y = pd.DataFrame(data=boston.target, columns=["target"])
    return train_test_split(X, y, test_size=0.2)


def run_mlbase_sklearn_classification(context):
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = get_dataset()
    model = apply_mlrun(
        model, context, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    model.fit(X_train, y_train)


def run_mlbase_xgboost_regression(context):
    try:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        X_train, X_test, y_train, y_test = get_dataset(classification=False)
        model = apply_mlrun(
            model,
            context,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        model.fit(X_train, y_train)
    except ModuleNotFoundError:
        pass


def run_mlbase_lgbm_classification(context):
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier()
        X_train, X_test, y_train, y_test = get_dataset()
        model = apply_mlrun(
            model,
            context,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        model.fit(X_train, y_train)
    except ModuleNotFoundError:
        pass


def test_run_mlbase_sklearn_classification():
    sklearn_run = new_function().run(handler=run_mlbase_sklearn_classification)
    assert (sklearn_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert (
        sklearn_run.artifact("model").meta.to_dict()["model_file"]
    ) == "LogisticRegression.pkl"


@pytest.mark.skipif(_is_installed("xgboost"), reason="xgboost package missing")
def test_run_mlbase_xgboost_regression():
    xgb_run = new_function().run(handler=run_mlbase_xgboost_regression)
    assert (xgb_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert "confusion matrix" not in (
        xgb_run.artifact("model").meta.to_dict()["extra_data"]
    )
    assert (
        xgb_run.artifact("model").meta.to_dict()["model_file"]
    ) == "XGBRegressor.pkl"


@pytest.mark.skipif(_is_installed("lightgbm"), reason="missing packages")
def test_run_mlbase_lgbm_classification():
    lgbm_run = new_function().run(handler=run_mlbase_lgbm_classification)
    assert (lgbm_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert (
        lgbm_run.artifact("model").meta.to_dict()["model_file"]
    ) == "LGBMClassifier.pkl"
