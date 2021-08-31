import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from mlrun.frameworks.sklearn import apply_mlrun as apply_mlrun_sklearn
from mlrun.frameworks.xgboost import apply_mlrun as apply_mlrun_xgb
from mlrun import new_function

def get_dataset(classification=True):
    if classification:
        iris = load_iris()
        X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        y = pd.DataFrame(data=iris.target, columns=['species'])
    else:
        boston = load_boston()
        X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        y = pd.DataFrame(data=boston.target, columns=['target'])
    return train_test_split(X, y, test_size=0.2)


def run_mlbase_sklearn_classification(context):
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = get_dataset()
    model = apply_mlrun_sklearn(model, context, model_name='my_model_name', X_test=X_test, y_test=y_test)
    model.fit(X_train, y_train)


def run_mlbase_xgboost_regression(context):
    model = xgb.XGBRegressor()
    X_train, X_test, y_train, y_test = get_dataset(classification=False)
    model = apply_mlrun_xgb(model, context)
    model.fit(X_train, y_train)


def test_run_mlbase_sklearn_classification():
    sklearn_run = new_function().run(handler=run_mlbase_sklearn_classification)
    assert (sklearn_run.artifact('my_model_name').meta.to_dict()['metrics']['accuracy']) > 0
    assert (sklearn_run.artifact('my_model_name').meta.to_dict()['model_file']) == 'LogisticRegression.pkl'


def test_run_mlbase_xgboost_regression():
    xgb_run = new_function().run(handler=run_mlbase_xgboost_regression)
    assert xgb_run.artifact('train_set').meta.to_dict()['target_path'] == 'data/train_set.csv'
    assert (len(xgb_run.artifact('train_set').meta.to_dict()['schema']['fields'])) == 16
