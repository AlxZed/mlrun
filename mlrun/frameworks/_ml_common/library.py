from typing import Any
from enum import Enum
from plotly.figure_factory import create_annotated_heatmap
from typing import Dict
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_curve, roc_auc_score
from IPython.core.display import HTML, display
from mlrun.artifacts import Artifact
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

import inspect
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mlrun


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
    Compute confusion matrix to plot the accuracy of a classification with Plotly.
    """

    _ARTIFACT_NAME = "confusion_matrix"

    def __init__(
        self, labels=None, sample_weight=None, normalize=None, y_test=None, y_pred=None
    ):
        """
        :param labels: List of labels to index the matrix.
        :param sample_weight: Sample weights.
        :param normalize: Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
        :param y_test: Ground truth (correct) target values.
        :param y_pred: Estimated targets as returned by a classifier.
        """
        validate_numerical(y_test)
        validate_numerical(y_pred)

        # confusion_matrix() parameters
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize

        super(ConfusionMatrixPlan, self).__init__(y_test=y_test, y_pred=y_pred)

    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.
        :param stage: The current stage in the function's process.
        :return: True if the given stage fits the plan and False otherwise.
        """
        return stage == PlanStages.POST_EVALUATION

    def produce(self, y_test, y_pred, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the plotly artifact according to this plan.
        :param y_pred: Estimated targets as returned by a classifier.
        :param y_test: Ground truth (correct) target values.
        :return: The produced artifacts.
        """
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
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(ConfusionMatrixPlan, self).display()


class ROCCurves(Plan):
    """ """

    _ARTIFACT_NAME = "roc_curves"

    def __init__(
        self,
        model=None,
        X_test=None,
        y_test=None,
        pos_label=None,
        sample_weight=None,
        drop_intermediate=True,
        average="macro",
        max_fpr=None,
        multi_class="raise",
        labels=None,
    ):
        """

        :param model: a fitted model
        :param X_test: train dataset used to verified a fitted model.
        :param y_test: target dataset.
        :param pos_label: The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an error will be raised.
        :param sample_weight: Sample weights.
        :param drop_intermediate: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve.
        :param average: If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data
        :param max_fpr: If not None, the standardized partial AUC [2] over the range [0, max_fpr] is returned.
        :param multi_class: Only used for multiclass targets. Determines the type of configuration to use.
        :param labels: Only used for multiclass targets. List of labels that index the classes in y_score
        """

        validate_numerical(X_test)
        validate_numerical(y_test)

        self._pos_label = pos_label
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate
        self.average = average
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

        super(ROCCurves, self).__init__(model=model, X_test=X_test, y_test=y_test)

    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.
        :param stage: The current stage in the function's process.
        :return: True if the given stage fits the plan and False otherwise.
        """
        return stage == PlanStages.POST_FIT

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

                fpr, tpr, _ = roc_curve(
                    y_true,
                    y_score,
                    pos_label=self._pos_label,
                    sample_weight=self._sample_weight,
                    drop_intermediate=self._drop_intermediate,
                )

                auc_score = roc_auc_score(
                    y_true,
                    y_score,
                    average=self._average,
                    sample_weight=self._sample_weight,
                    max_fpr=self._max_fpr,
                    multi_class=self._multi_class,
                    labels=self._labels,
                )

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
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(ROCCurves, self).display()


class FeatureImportance(Plan):
    """ """

    _ARTIFACT_NAME = "feature_importance"

    def __init__(
        self,
        model=None,
        X_train=None,
    ):
        """
        :param model: any model pre-fit or post-fit.
        :param X_train: train dataset.
        """
        validate_numerical(X_train)

        super(FeatureImportance, self).__init__(model=model, X_train=X_train)

    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.
        :param stage: The current stage in the function's process.
        :return: True if the given stage fits the plan and False otherwise.
        """
        return stage == PlanStages.PRE_FIT

    def produce(self, model, X_train, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):

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
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "This model cannot be used for FeatureImportance plotting."
            )

    def display(self):
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(FeatureImportance, self).display()


class LearningCurves(Plan):
    """
    Determines cross-validated training and test scores for different training set sizes.
    """

    _ARTIFACT_NAME = "learning_curves"

    def __init__(
        self,
        model=None,
        X_train=None,
        y_train=None,
        cv=3,
        groups=None,
        train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.0]),
        scoring=None,
        exploit_incremental_learning=False,
        n_jobs=None,
        pre_dispatch="all",
        verbose=0,
        shuffle=False,
        random_state=None,
        return_times=True,
        fit_params=None,
    ):
        """

        :param model: a fitted model.
        :param X_train: training dataset.
        :param y_train: target dataset.
        :param cv: Determines the cross-validation splitting strategy.
        :param groups: Group labels for the samples used while splitting the dataset into train/test set.
        :param train_sizes: Relative or absolute numbers of training examples that will be used to generate the learning curve.
        :param scoring: A str (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
        :param exploit_incremental_learning: If the estimator supports incremental learning, this will be used to speed up fitting for different training set sizes.
        :param n_jobs: Number of jobs to run in parallel.
        :param pre_dispatch: Number of predispatched jobs for parallel execution (default is all).
        :param verbose: Controls the verbosity: the higher, the more messages.
        :param shuffle: Whether to shuffle training data before taking prefixes of it based on``train_sizes``.
        :param random_state: Used when shuffle is True. Pass an int for reproducible output across multiple function calls.
        :param return_times: Value to assign to the score if an error occurs in estimator fitting.
        :param fit_params: Parameters to pass to the fit method of the estimator.
        """
        validate_numerical(X_train)
        validate_numerical(y_train)

        # learning_curve() params
        self._groups = groups
        self._cv = cv
        self._train_sizes = train_sizes
        self._scoring = scoring
        self._exploit_incremental_learning = exploit_incremental_learning
        self._n_jobs = n_jobs
        self._pre_dispatch = pre_dispatch
        self._verbose = verbose
        self._shuffle = shuffle
        self._random_state = random_state
        self._return_times = return_times
        self._fit_params = fit_params

        super(LearningCurves, self).__init__(
            model=model, X_train=X_train, y_train=y_train
        )

    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.
        :param stage: The current stage in the function's process.
        :return: True if the given stage fits the plan and False otherwise.
        """
        return stage == PlanStages.POST_FIT

    def produce(self, model, X_train, y_train, **kwargs) -> Dict[str, PlotlyArtifact]:

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            X_train,
            y_train.values.ravel(),
            groups=self._groups,
            cv=self._cv,
            train_sizes=self._train_sizes,
            scoring=self._scoring,
            _exploit_incremental_learning=self._exploit_incremental_learning,
            n_jobs=self._n_jobs,
            pre_dispatch=self._pre_dispatch,
            verbose=self._verbose,
            shuffle=self._shuffle,
            random_state=self._random_state,
            return_times=self._return_times,
            fit_params=self._fit_params,
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
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(LearningCurves, self).display()


class CalibrationCurve(Plan):
    """
    Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier, and discretize the [0, 1] interval into bins.
    """

    _ARTIFACT_NAME = "calibration_curve"

    def __init__(
        self,
        model=None,
        X_test=None,
        y_test=None,
        normalize=False,
        n_bins=5,
        strategy="uniform",
    ):
        """

        :param model: a fitted model with attributep redict_proba.
        :param X_test: train dataset used to verified a fitted model.
        :param y_test: target dataset.
        :param normalize:Whether y_prob needs to be normalized into the [0, 1] interval, i.e. is not a proper probability.
        :param n_bins: Number of bins to discretize the [0, 1] interval.
        :param strategy: Strategy used to define the widths of the bins.
        """
        validate_numerical(X_test)
        validate_numerical(y_test)

        # calibration_curve() parameters
        self._normalize = normalize
        self._n_bins = n_bins
        self._strategy = strategy

        super(CalibrationCurve, self).__init__(
            model=model, X_test=X_test, y_test=y_test
        )

    def is_ready(self, stage: PlanStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.
        :param stage: The current stage in the function's process.
        :return: True if the given stage fits the plan and False otherwise.
        """
        return stage == PlanStages.POST_FIT

    def produce(self, model, X_test, y_test, **kwargs) -> Dict[str, PlotlyArtifact]:

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(
                y_test,
                probs,
                n_bins=self._n_bins,
                normalize=self._normalize,
                strategy=self._strategy,
            )

            fig = go.Figure(
                data=[go.Scatter(x=prob_true, y=prob_pred)],
                layout=dict(title=dict(text="Calibration Curve")),
            )

            # add custom xaxis title
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=14),
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="prob_true",
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
                    text="prob_pred",
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
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "This model cannot be used for CalibrationCurve plotting."
            )

    def display(self):
        """
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(CalibrationCurve, self).display()
