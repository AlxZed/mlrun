""
Description:
__init__ function of Xgboost-autologger. Will be extended and contain multiple Xgboost-specific functions.
"""

import mlrun
from mlrun.frameworks.xgboost.mlrun_interface import MLBaseMLRunInterface
from mlrun.frameworks._common.pkl_model_server import PklModelServer

# Temporary placeholder, XGBModelServer may deviate from PklModelServer in upcoming versions.
XGBModelServer = PklModelServer
        
def apply_mlrun(
        model,
        context: mlrun.MLClientCtx = None,
        **kwargs):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    Usage Example: model = apply_mlrun_xgb(model, context, X_train=X_train,
                        y_train=y_train, X_test=X_test, y_test=y_test)
                
    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx('mlrun_xgb')

    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, kwargs)
    return model
