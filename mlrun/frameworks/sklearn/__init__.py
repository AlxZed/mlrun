"""
Description:
__init__ function of sklearn-autologger. Will be extended and contain multiple Sklearn-specific functions.
"""

import mlrun
from mlrun.frameworks._common.pkl_model_server import PickleModelServer
from plans_manager import ArtifactsPlansManager
from library import SkLearnArtifactLibrary

# Temporary placeholder, SklearnModelServer may deviate from PklModelServer in upcoming versions.
SklearnModelServer = PickleModelServer


def apply_mlrun(
    model,
    context: mlrun.MLClientCtx = None,
    model_name=None,
    generate_test_set=True,
    feature_vector = None,
    here=True,
    artifact_list=None,
    **kwargs
):

    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_sklearn")

    if feature_vector and hasattr(feature_vector, "uri"): 
         kwargs["feature_vector"]=feature_vector.uri
            
    # Detect model library
    is_sklearn, is_xgb, is_lgbm = detect_model_library(model)
    
    # Assign artifact_list
    if is_sklearn:
        artifact_list=artifact_list if artifact_list is not None else SkLearnArtifactLibrary.default()
    elif is_xgb:
        artifact_list=artifact_list if artifact_list is not None else XgbArtifactLibrary.default()
    elif is_lgbm:
        artifact_list=artifact_list if artifact_list is not None else LgbmArtifactLibrary.default()
    
    #
    plans_manager = ArtifactsPlansManager(plans=artifact_list)
    
    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model=model, context=context, model_name=model_name, plans_manager=plans_manager, apply_args=kwargs)
    return model
