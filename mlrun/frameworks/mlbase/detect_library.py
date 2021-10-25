def detect_model_library(model) -> bool:
    """
    Identifies which of the main three Machine Learning library a model is using

    :param model:           fitted or unfitted model from xgboost, sklearn or lgbm.

    :return is_sklearn:     a boolean value equivalent to sklearn.
    :return is_xgb:         a boolean value equivalent to sklearn
    :return is_lgbm:        a boolean value equivalent to sklearn
    """

    is_sklearn = False
    is_xgb = False
    is_lgbm = False

    model_library = str(model.__class__.__module__).split('.')[0]

    if model_library == 'sklearn':
        is_sklearn = True

    elif model_library == 'xgboost':
        is_xgb = True

    elif model_library == 'lightgbm':
        is_lgbm = True

    return is_sklearn, is_xgb, is_lgbm
