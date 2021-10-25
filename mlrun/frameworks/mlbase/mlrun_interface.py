import pandas as pd
import numpy as np

from cloudpickle import dumps
from mlrun.frameworks._common import MLRunInterface
from plan import ProductionStages

class MLBaseMLRunInterface(MLRunInterface):
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @classmethod
    def add_interface(cls, model, context, model_name, plans_manager, apply_args={}, *args, **kwargs):

        
        # Validate artifacts to be produced
        plans_manager.validate()
        
        # Wrap the fit method:
        def fit_wrapper(fit_method, **kwargs):
            def wrapper(*args, **kwargs):
                
                
                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)

                # Post fit
                _post_fit(*args, **kwargs)

            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit, **kwargs))
        
        
        def _post_fit(*args, **kwargs):
            context.set_label("class", str(model.__class__.__name__))
            
            # Identify splits and build test set
            X_train = args[0]
            y_train = args[1]

            if isinstance(X_train, np.ndarray): X_train=pd.DataFrame(X_train)
            if isinstance(y_train, np.ndarray): y_train=pd.DataFrame(y_train)

            train_set = pd.concat([X_train, y_train], axis=1)
            train_set.reset_index(drop=True, inplace=True)
    
            plans_manager.generate(model=model, context=context, mystage=ProductionStages.POST_FIT, apply_args=apply_args, **kwargs)

                
            if apply_args.get("X_test") is not None and apply_args.get("y_test") is not None:
                
                # Identify splits and build test set
                X_test=pd.DataFrame(apply_args["X_test"]) if isinstance(apply_args["X_test"], np.ndarray) else apply_args["X_test"]
                y_test=pd.DataFrame(apply_args["y_test"]) if isinstance(apply_args["y_test"], np.ndarray) else apply_args["y_test"]

                test_set = pd.concat([X_test, y_test], axis=1)
                test_set.reset_index(drop=True, inplace=True)
                

                if apply_args.get("generate_test_set"):
                    # Log test dataset
                    context.log_dataset(
                        "test_set",
                        df=test_set,
                        format="parquet",
                        index=False,
                        labels={"data-type": "held-out"},
                        artifact_path=context.artifact_subpath("data"),
                    )

            # Log fitted model and metrics
            label_column = (y_train.name if isinstance(y_train, pd.Series) else y_train.columns.to_list())
            
            context.log_model(
                model_name or "model",
                db_key=model_name,
                body=dumps(model),
                artifact_path=context.artifact_subpath("models"),
                framework=f"{str(model.__module__).split('.')[0]}",
                algorithm=str(model.__class__.__name__),
                model_file=f"{str(model.__class__.__name__)}.pkl",
                metrics=context.results,
                format="pkl",
                training_set=train_set,
                label_column=label_column,
#                 extra_data=eval_metrics,
            )
