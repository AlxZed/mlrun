def detect_model_library(model) -> bool:
    """
    Identifies which of the main three Machine Learning library a model is using.

    :param model:           fitted or unfitted model from xgboost, sklearn or lgbm.

    :return is_sklearn:     a boolean equivalent to sklearn.
    :return is_xgb:         a boolean equivalent to xgboost.
    :return is_lgbm:        a boolean equivalent to lightgbm.
    """
    
    # Parse library from model class
    model_library = str(model.__class__.__module__).split('.')[0]
    
    # Turn the according boolean flags
    is_sklearn = True if model_library == 'sklearn' else False
    is_xgb = True if model_library == 'xgboost' else False
    is_lgbm = True if model_library == 'lightgbm' else False

    return is_sklearn, is_xgb, is_lgbm
