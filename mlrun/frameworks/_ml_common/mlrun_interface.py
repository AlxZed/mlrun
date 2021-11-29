from typing import List

import numpy as np
import pandas as pd

from .model_handler import MLModelHandler
from .plots import eval_model_v2


class MLMLRunInterface:
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @staticmethod
    def numpy_array_to_df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Transform and rename numpy arrays to DataFrames.
        :param x: an np array containing parameters
        :param y: an np array containing y targets
        :return: x,y as dataframes
        """
        df_x = pd.DataFrame(x, columns=["param_%d" % (i + 1) for i in range(x.shape[1])])

        # if y has one column (100,)
        if len(y.shape) == 1:
            df_y = pd.DataFrame(y, columns=["y"])

        # if y has multiple columns (100,5)
        elif len(y.shape) > 1:
            df_y = pd.DataFrame(y, columns=["y_%d" % (i + 1) for i in range(y.shape[1])])
        return df_x, df_y

    @staticmethod
    def merge_dataframes(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two dataframes while making sure their indices are aligned.
        :param x: train data as a pd.DataFrame
        :param y: label column as a pd.DataFrame
        :return full_df: merged dataframe of x and y
        """
        # Checking if indexes are aligned
        assert set(x.index.tolist()) == set(
            y.index.tolist()
        ), "Pandas Dataframe indexes are not equal.."

        # Merge X and y for logging of the train set
        full_df = pd.concat([x, y], axis=1)
        full_df.reset_index(drop=True, inplace=True)

        return full_df

    @classmethod
    def add_interface(
        cls,
        model_handler: MLModelHandler,
        context,
        model_name,
        data={},
        *args,
        **kwargs
    ):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.
        :param model:       The model to wrap.
        :param context:     MLRun context to work with. If no context is given it will be retrieved via
                            'mlrun.get_or_create_ctx(None)'
        :param model_name:  name under whcih the model will be saved within the databse.
        :param data:        Optional: The train_test_split X_train, X_test, y_train, y_test can be passed,
                                      or the test data X_test, y_test can be passed.
        :return: The wrapped model.
        """
        model = model_handler.model

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
            eval_metrics = None
            context.set_label("class", str(model.__class__.__name__))

            # Get passed X,y from model.fit(X,y)
            x_train, y_train = args[0], args[1]

            # np.array -> Dataframe
            if isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray):
                x_train, y_train = MLMLRunInterface.numpy_array_to_df(x_train, y_train)

            # Merge X and y for logging of the test set
            train_set = MLMLRunInterface.merge_dataframes(x_train, y_train)

            if data.get("X_test") is not None and data.get("y_test") is not None:
                # Identify splits and build test set
                x_test, y_test = data["X_test"], data["y_test"]

                # np.array -> Dataframe
                if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                    x_test, y_test = MLMLRunInterface.numpy_array_to_df(x_test, y_test)

                # Merge X and y for logging of the test set
                test_set = MLMLRunInterface.merge_dataframes(x_test, y_test)

                # Evaluate model results and get the evaluation metrics
                eval_metrics = eval_model_v2(context, x_test, y_test, model)

                if data.get("generate_test_set"):
                    # Log test dataset
                    context.log_dataset(
                        "test_set",
                        df=test_set,
                        format="parquet",
                        index=False,
                        labels={"data-type": "held-out"},
                        artifact_path=context.artifact_subpath("data"),
                    )

            # Identify label column
            label_column = None  # type: List[str]
            if isinstance(y_train, pd.DataFrame):
                label_column = y_train.columns.to_list()
            elif isinstance(y_train, pd.Series):
                if y_train.name is not None:
                    label_column = [str(y_train.name)]
                else:
                    raise ValueError("No column name for y_train was specified")

            model_handler.log(
                algorithm=str(model.__class__.__name__),
                training_set=train_set,
                label_column=label_column,
                extra_data=eval_metrics,
                artifacts=eval_metrics,
                metrics=context.results,
            )
