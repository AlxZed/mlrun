import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plan import ProductionStages
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from itertools import cycle
from typing import List
from mlrun.artifacts import PlotArtifact
from plan import ArtifactPlan


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

    def df_blob(df):
        """
        """
        return bytes(df.to_csv(index=False), encoding="utf-8")

    def precision_recall_multi(ytest_b, yprob, labels, scoring="micro"):
        n_classes = len(labels)

        precision = dict()
        recall = dict()
        avg_prec = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = metrics.precision_recall_curve(
                ytest_b[:, i], yprob[:, i]
            )
            avg_prec[i] = metrics.average_precision_score(ytest_b[:, i], yprob[:, i])
        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(
            ytest_b.ravel(), yprob.ravel()
        )
        avg_prec["micro"] = metrics.average_precision_score(ytest_b, yprob, average="micro")
        ap_micro = avg_prec["micro"]
        # model_metrics.update({'precision-micro-avg-classes': ap_micro})

        # gcf_clear(plt)
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append("iso-f1 curves")
        (l,) = plt.plot(recall["micro"], precision["micro"], color="gold", lw=10)
        lines.append(l)
        labels.append(f"micro-average precision-recall (area = {ap_micro:0.2f})")

        for i, color in zip(range(n_classes), colors):
            (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append(f"precision-recall for class {i} (area = {avg_prec[i]:0.2f})")

        # fig = plt.gcf()
        # fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision recall - multiclass")
        plt.legend(lines, labels, loc=(0, -0.41), prop=dict(size=10))

        return PlotArtifact(
            "precision-recall-multiclass",
            body=plt.gcf(),
            title="Multiclass Precision Recall",
        )

    def check_multiclass(self, ytest):
        """
        """
        if isinstance(ytest, (pd.DataFrame, pd.Series)):
            ytest = ytest.to_numpy()
        if isinstance(ytest, np.ndarray):
            self._unique_labels = np.unique(ytest)
        elif isinstance(ytest, list):
            self._unique_labels = set(ytest)
        return True if len(self._unique_labels) > 2 else False

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

    class ROCCurves(ArtifactPlan):
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

