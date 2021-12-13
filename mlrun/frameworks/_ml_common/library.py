import plotly.figure_factory as ff
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from typing import List
from .._ml_common.plan import ArtifactPlan, ProductionStages
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from IPython.core.display import HTML, display
from mlrun.artifacts import Artifact
from sklearn.model_selection import learning_curve


class ArtifactLibrary(ABC):
    """
    An abstract class for an artifacts library. Each framework should have an artifacts library for knowing what
    artifacts can be produced and their configurations. The default method must be implemented for when the user do not
    pass any plans.
    """

    @classmethod
    @abstractmethod
    def default(cls) -> List[ArtifactPlan]:
        """
        Get the default artifacts plans list of this framework's library.

        :return: The default artifacts plans list.
        """
        pass


class ROCCurve(ArtifactPlan):
    """
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_EVALUATION

    @abstractmethod
    def produce(self, model, context, apply_args, plots_artifact_path="", **kwargs):
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(apply_args['X_test'])

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(apply_args['y_test'], columns=model.classes_)

        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))


class ConfusionMatrix(ArtifactPlan):

    def __init__(
            self,
            labels=None,
            sample_weight=None,
            normalize=None,
            **kwargs):

        # confusion_matrix() parameters
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize

        self._kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        """
        Validate this plan has the required data to produce its artifact.

        :raise ValueError: In case this plan is missing information in order to produce its artifact.
        """
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_EVALUATION

    @abstractmethod
    def produce(self, apply_args, **kwargs) -> Artifact:
        """
        Produce the artifact according to this plan.

        :return: The produced artifact.
        """
        y_test = apply_args['y_test']
        y_pred = apply_args['y_pred']

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
        )

        x = np.sort(y_test[y_test.columns[0]].unique()).tolist()

        # set up figure
        fig = ff.create_annotated_heatmap(cm, x=x, y=x, annotation_text=cm.astype(str), colorscale='Blues')

        # add title
        fig.update_layout(title_text='Confusion matrix',
                          )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.2,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=100, l=100), width=500, height=500)

        # add colorbar
        fig['data'][0]['showscale'] = True
        fig['layout']['yaxis']['autorange'] = "reversed"

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))


class FeatureImportance(ArtifactPlan):
    """
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        """
        Validate this plan has the required data to produce its artifact.

        :raise ValueError: In case this plan is missing information in order to produce its artifact.
        """
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_FIT

    @abstractmethod
    def produce(self, model, apply_args, *args, **kwargs) -> Artifact:
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

        df = pd.DataFrame({'features': apply_args['X_train'].columns, 'feature_importance': importance_score}).sort_values(
            by="feature_importance", ascending=False
        )

        fig = go.Figure([go.Bar(x=df['feature_importance'], y=df['features'], orientation='h')])

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))


class LearningCurves(ArtifactPlan):
    """
    SkLearn Learning Curves
    """

    def __init__(self,
                 cv=3,
                 **kwargs):

        # Learning Curves parameters
        self._cv = cv

        # Other
        self.kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        """
        Validate this plan has the required data to produce its artifact.

        :raise ValueError: In case this plan is missing information in order to produce its artifact.
        """
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_FIT

    @abstractmethod
    def produce(self, model, apply_args, *args, **kwargs) -> Artifact:
        """
        Produce the artifact according to this plan.

        :return: The produced artifact.
        """

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, apply_args['X_train'],
                                                                              apply_args['y_train'].values.ravel(), cv=self._cv,
                                                                              return_times=True)

        fig = go.Figure(
            data=[go.Scatter(x=train_sizes.tolist(), y=np.mean(train_scores, axis=1))],
            layout=dict(title=dict(text="Learning Curves"))
        )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Train Size",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Score",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=100, l=100), width=800, height=500)

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))


class SkLearnArtifactLibrary(ArtifactLibrary):
    @classmethod
    def default(cls) -> List[ArtifactPlan]:
        return [ArtifactLibrary.ConfusionMatrix(cmap='Blues')]

    @staticmethod
    def confusion_matrix(a: int = 9, b: float = 9.9, c: str = "9"):
        return ConfusionMatrixPlan(a=a, b=b, c=c)


class XGBArtifactLibrary(ArtifactLibrary):
    @classmethod
    def default(cls) -> List[ArtifactPlan]:
        return [ArtifactLibrary.ConfusionMatrix(cmap='Blues')]


class LGBMArtifactLibrary(ArtifactLibrary):
    @classmethod
    def default(cls) -> List[ArtifactPlan]:
        return [ArtifactLibrary.ConfusionMatrix(cmap='Blues')]
