import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from typing import List
from mlrun.artifacts import PlotArtifact
from .._ml_common.plan import ArtifactPlan, ProductionStages


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


class ConfusionMatrix(ArtifactPlan):
    """
    """

    def __init__(
            self,
            labels=None,
            sample_weight=None,
            normalize=None,
            display_labels=None,
            include_values=True,
            xticks_rotation="horizontal",
            values_format=None,
            cmap="Blues",
            ax=None,
            **kwargs):

        # confusion_matrix() parameters
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize

        # plot_confusion_matrix() parameters
        self._cmap = cmap
        self._values_format = values_format
        self._display_labels = display_labels
        self._include_values = include_values
        self._xticks_rotation = xticks_rotation
        self._ax = ax

        # other
        self._kwargs = kwargs
        self._extra_data = {}

    def validate(self, *args, **kwargs):
        pass

    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_FIT

    def produce(self, model, context, apply_args, plots_artifact_path="", **kwargs):
        # generate confusion matrix
        cm = confusion_matrix(
            apply_args['y_test'],
            apply_args['y_pred'],
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
        )

        # Create a dataframe from cmatrix
        df = pd.DataFrame(data=cm)

        # Add the dataframe to extra_data
        self._extra_data["confusion_matrix_table.csv"] = ArtifactLibrary.df_blob(df)

        cmd = metrics.plot_confusion_matrix(
            model,
            apply_args['X_test'],
            apply_args['y_test'],
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
            display_labels=self._display_labels,
            include_values=self._include_values,
            xticks_rotation=self._xticks_rotation,
            values_format=self._values_format,
            cmap=self._cmap,
            ax=self._ax,
        )

        self._extra_data["confusion matrix"] = context.log_artifact(
            PlotArtifact(
                "confusion-matrix",
                body=cmd.figure_,
                title="Confusion Matrix - Normalized Plot",
            ),
            artifact_path=plots_artifact_path or context.artifact_subpath("plots"),
            db_key=False,
        )

        return self._extra_data

    class FeatureImportance(ArtifactPlan):
        """
        """

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._extra_data = {}

        def validate(self, *args, **kwargs):
            pass

        def is_ready(self, stage: ProductionStages) -> bool:
            return stage == ProductionStages.POST_FIT

        def produce(self, model, context, apply_args, plots_artifact_path="", **kwargs):

            """Display estimated feature importances
            Only works for models with attribute 'feature_importances_`
            Feature importances are provided by the fitted attribute feature_importances_ and they are computed
            as the mean and standard deviation of accumulation of the impurity decrease within each tree.

            :param model:       fitted model
            :param header:      feature labels
            """

            # Tree-based feature importance
            if hasattr(model, "feature_importances_"):
                importance_score = zip(model.feature_importances_, apply_args['X_train'].columns)

            # Coefficient-based importance
            elif hasattr(model, "coef_"):
                importance_score = zip(model.coef_, apply_args['X_train'].columns)

            feature_imp = pd.DataFrame(sorted(importance_score), columns=["Importance score", "feature"]).sort_values(
                by="Importance score", ascending=False
            )

            plt.figure()
            sns.barplot(x="Importance score", y="feature", data=feature_imp)
            plt.title("features")
            plt.tight_layout()

            self._extra_data["feature importance"] = context.log_artifact(
                PlotArtifact(
                    "feature-importances", body=plt.gcf(), title="Feature Importances"
                ),
                artifact_path=plots_artifact_path or context.artifact_subpath("plots"),
                db_key=False, )

            return self._extra_data, feature_imp


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
