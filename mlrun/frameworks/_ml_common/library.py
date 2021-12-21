from typing import Dict, Any
from enum import Enum

import inspect


from plotly.figure_factory import create_annotated_heatmap
from typing import Dict
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import mlrun
from sklearn.metrics import roc_curve, roc_auc_score
from IPython.core.display import HTML, display
from mlrun.artifacts import Artifact
from sklearn.model_selection import learning_curve


class PlanStages(Enum):
    """
    Stages for a plan to be produced.
    """

    PRE_FIT = "pre_fit"
    POST_FIT = "post_fit"
    PRE_EVALUATION = "pre_evaluation"
    POST_EVALUATION = "post_evaluation"


class Plan(ABC):
    """
    An abstract class for describing a plan. A plan is used to produce artifact in a given time of a function according
    to its configuration.
    """

    def __init__(self, auto_produce: bool = True, **produce_arguments):
        """
        Initialize a new plan. The plan will be automatically produced if all of the required arguments to the produce
        method are given.

        :param auto_produce:      Whether to automatically produce the artifact if all of the required arguments are
                                  given. Defaulted to True.
        :param produce_arguments: The provided arguments to the produce method in kwargs style.
        """
        # Set the artifacts dictionary:
        self._artifacts = {}  # type: Dict[str, Artifact]

        # Check if the plan should be produced, if so call produce:
        if auto_produce and self._is_producible(given_parameters=produce_arguments):
            self.produce(**produce_arguments)

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        Get the plan's produced artifacts.

        :return: The plan's artifacts.
        """
        return self._artifacts

    @abstractmethod
    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.

        :param stage: The current stage in the function's process.

        :return: True if the given stage fits the plan and False otherwise.
        """
        pass

    @abstractmethod
    def produce(self, *args, **kwargs) -> Dict[str, Artifact]:
        """
        Produce the artifact according to this plan.

        :return: The produced artifacts.
        """
        pass

    def log(self, context: mlrun.MLClientCtx):
        """
        Log the artifacts in this plan to the given context.

        :param context: A MLRun context to log with.
        """
        for artifact_name, artifact_object in self._artifacts.items():
            context.log_artifact(artifact_object)

    def display(self):
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        print(repr(self))

    def _is_producible(self, given_parameters: Dict[str, Any]) -> bool:
        """
        Check if the plan can call its 'produce' method with the given parameters. If all of the required parameters are
        given, True is returned and Otherwise, False.

        :param given_parameters: The given parameters to check.

        :return: True if the plan is producible and False if not.
        """
        # Get this plan's produce method required parameters:
        required_parameters = [
            parameter_name
            for parameter_name, parameter_object in inspect.signature(
                self.produce
            ).parameters.items()
            if parameter_object.default == parameter_object.empty
            and parameter_object.name != "kwargs"
        ]

        # Parse the given parameters into a list of the actual given (not None) parameters:
        given_parameters = [
            parameter_name
            for parameter_name, parameter_value in given_parameters.items()
            if parameter_value is not None
        ]

        # Return True only if all of the required parameters are available from the given parameters:
        return all(
            [
                required_parameter in given_parameters
                for required_parameter in required_parameters
            ]
        )

    def _repr_pretty_(self, p, cycle: bool):
        """
        A pretty representation of the plan. Will be called by the IPython kernel. This method will call the plan's
        display method.

        :param p:     A RepresentationPrinter instance.
        :param cycle: If a cycle is detected to prevent infinite loop.
        """
        self.display()


class PlotlyArtifact(Artifact):
    """
    Plotly artifact is an artifact for saving Plotly generated figures. They will be stored in a html format.
    """

    kind = "plotly"

    def __init__(
        self,
        figure,
        key: str = None,
        target_path: str = None,
    ):
        """
        Initialize a Plotly artifact with the given figure.

        :param figure:      Plotly figure ('plotly.graph_objs.Figure' object) to save as an artifact.
        :param key:         Key for the artifact to be stored in the database.
        :param target_path: Path to save the artifact.
        """
        super().__init__(key=key, target_path=target_path, viewer="plotly")

        # Validate input:
        try:
            from plotly.graph_objs import Figure
        except (ModuleNotFoundError, ImportError) as Error:
            raise Error(
                "Using 'PlotlyArtifact' requires plotly package. Use pip install mlrun[plotly] to install it."
            )
        if not isinstance(figure, Figure):
            raise ValueError(
                "PlotlyArtifact requires the figure parameter to be a "
                "'plotly.graph_objs.Figure' but received '{}'".format(type(figure))
            )

        # Continue initializing the plotly artifact:
        self._figure = figure
        self.format = "html"

    def get_body(self):
        """
        Get the artifact's body - the Plotly figure's html code.

        :return: The figure's html code.
        """
        return self._figure.to_html()


def validate_numerical(dataset):
    """
    Validate the input and the data passed to the Artifact maker.

    :raise ValueError: In case this plan is missing information in order to produce its artifact.
    """
    if isinstance(dataset, (pd.DataFrame, np.ndarray, pd.Series)) is False:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "dataframe must be pd.DataFrame, np.array, pd.Series"
        )

    # When data passed is a DataFrame
    if (
        isinstance(dataset, (pd.DataFrame))
        and all(is_numeric_dtype(dataset[col]) for col in dataset.columns) is False
    ):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "dataframe must be numerical (float or int)"
        )

    # When data passed is a Series
    if isinstance(dataset, (pd.Series)) and is_numeric_dtype(dataset) is False:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "dataframe must be numerical (float or int)"
        )

    # When data passed is a np.array
    if isinstance(dataset, (np.ndarray)) and dataset.dtype not in [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "dataframe must be numerical (float or int)"
        )


class ConfusionMatrixPlan(Plan):
    """
    Compute confusion matrix to evaluate the accuracy of a classification with Plotly.
    """

    _ARTIFACT_NAME = "confusion_matrix"

    def __init__(
        self, labels=None, sample_weight=None, normalize=None, y_test=None, y_pred=None
    ):
        validate_numerical(y_test)
        validate_numerical(y_pred)

        # confusion_matrix() parameters
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize

        super(ConfusionMatrixPlan, self).__init__(y_test=y_test, y_pred=y_pred)

    def is_ready(self, stage: PlanStages) -> bool:
        return stage == PlanStages.POST_EVALUATION

    def produce(self, y_test, y_pred, **kwargs) -> Dict[str, PlotlyArtifact]:

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
        )

        x = np.sort(y_test[y_test.columns[0]].unique()).tolist()

        # set up figure
        figure = create_annotated_heatmap(
            cm, x=x, y=x, annotation_text=cm.astype(str), colorscale="Blues"
        )

        # add title
        figure.update_layout(
            title_text="Confusion matrix",
        )

        # add custom xaxis title
        figure.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Predicted value",
                xref="paper",
                yref="paper",
            )
        )

        # add custom yaxis title
        figure.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.2,
                y=0.5,
                showarrow=False,
                text="Real value",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        figure.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

        # adjust margins to make room for yaxis title
        figure.update_layout(margin=dict(t=100, l=100), width=500, height=500)

        # add colorbar
        figure["data"][0]["showscale"] = True
        figure["layout"]["yaxis"]["autorange"] = "reversed"

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=figure, key=self._ARTIFACT_NAME
        )
        return self._artifacts

    def display(self):
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(ConfusionMatrixPlan, self).display()


class ROCCurves(Plan):
    _ARTIFACT_NAME = "roc_curves"

    def __init__(self, model=None, X_test=None, y_test=None):
        validate_numerical(X_test)
        validate_numerical(y_test)

        super(ROCCurves, self).__init__(model=model, X_test=X_test, y_test=y_test)

    def is_ready(self, stage: PlanStages) -> bool:
        return stage == PlanStages.POST_EVALUATION

    def produce(self, model, X_test, y_test, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)

            # One hot encode the labels in order to plot them
            y_onehot = pd.get_dummies(y_test, columns=y_test.columns.to_list())

            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain="domain"),
                width=700,
                height=500,
            )

            # Creating an html rendering of the plot
            self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
                figure=fig, key=self._ARTIFACT_NAME
            )
            return self._artifacts

    def display(self):
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(ROCCurves, self).display()


class FeatureImportance(Plan):
    _ARTIFACT_NAME = "feature_importance"

    def __init__(
        self,
        model=None,
        X_train=None,
    ):
        validate_numerical(X_train)

        super(FeatureImportance, self).__init__(model=model, X_train=X_train)

    def is_ready(self, stage: PlanStages) -> bool:
        return stage == PlanStages.POST_FIT

    def produce(self, model, X_train, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        # Tree-based feature importance
        if hasattr(model, "feature_importances_"):
            importance_score = model.feature_importances_

        # Coefficient-based importance|
        elif hasattr(model, "coef_"):
            importance_score = model.coef_[0]

        df = pd.DataFrame(
            {
                "features": X_train.columns,
                "feature_importance": importance_score,
            }
        ).sort_values(by="feature_importance", ascending=False)

        fig = go.Figure(
            [go.Bar(x=df["feature_importance"], y=df["features"], orientation="h")]
        )

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=fig, key=self._ARTIFACT_NAME
        )
        return self._artifacts

    def display(self):
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(FeatureImportance, self).display()


class LearningCurves(Plan):
    _ARTIFACT_NAME = "learning_curves"

    def __init__(
        self,
        model=None,
        X_train=None,
        y_train=None,
        cv=3,
    ):
        self._cv = cv
        validate_numerical(X_train)
        validate_numerical(y_train)

        super(LearningCurves, self).__init__(
            model=model, X_train=X_train, y_train=y_train
        )

    def is_ready(self, stage: PlanStages) -> bool:
        return stage == PlanStages.POST_FIT

    def produce(self, model, X_train, y_train, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            X_train,
            y_train.values.ravel(),
            cv=self._cv,
            return_times=True,
        )

        fig = go.Figure(
            data=[go.Scatter(x=train_sizes.tolist(), y=np.mean(train_scores, axis=1))],
            layout=dict(title=dict(text="Learning Curves")),
        )

        # add custom xaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Train Size",
                xref="paper",
                yref="paper",
            )
        )

        # add custom yaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.1,
                y=0.5,
                showarrow=False,
                text="Score",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=100, l=100), width=800, height=500)

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=fig, key=self._ARTIFACT_NAME
        )
        return self._artifacts

    def display(self):
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(LearningCurves, self).display()
