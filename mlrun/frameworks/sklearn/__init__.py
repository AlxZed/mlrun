'''
__init__ function of sklearn-autologger. Will be extended and will contain multiple Sklearn-specific functions.
'''

import mlrun
from mlrun.frameworks.mlbase.mlrun_interface import MLBaseMLRunInterface
from mlrun.frameworks._common.pkl_model_server import PklModelServer

SklearnModelServer = PklModelServer
        
def apply_mlrun(
        model,
        context: mlrun.MLClientCtx = None,
        **kwargs):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.
    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx('mlrun_sklearn')
         
    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, kwargs)
    return model
