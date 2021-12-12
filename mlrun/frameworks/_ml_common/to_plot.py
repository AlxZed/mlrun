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

        def produce(self, model,
                context,
                y_labels,
                y_probs,
                key="roc",
                plots_dir: str = "plots",
                fmt="png",
                fpr_label: str = "false positive rate",
                tpr_label: str = "true positive rate",
                title: str = "roc curve",
                legend_loc: str = "best",
                clear: bool = True,
        ):
            """plot roc curves
            **legacy version please deprecate in functions and demos**
            :param context:      the function context
            :param y_labels:     ground truth labels, hot encoded for multiclass
            :param y_probs:      model prediction probabilities
            :param key:          ("roc") key of plot in artifact store
            :param plots_dir:    ("plots") destination folder relative path to artifact path
            :param fmt:          ("png") plot format
            :param fpr_label:    ("false positive rate") x-axis labels
            :param tpr_label:    ("true positive rate") y-axis labels
            :param title:        ("roc curve") title of plot
            :param legend_loc:   ("best") location of plot legend
            :param clear:        (True) clear the matplotlib figure before drawing
            """

            # draw 45 degree line
            plt.plot([0, 1], [0, 1], "k--")

            # labelling
            plt.xlabel(fpr_label)
            plt.ylabel(tpr_label)
            plt.title(title)
            plt.legend(loc=legend_loc)

            # single ROC or multiple
            if y_labels.shape[1] > 1:

                # data accumulators by class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(y_labels[:, :-1].shape[1]):
                    fpr[i], tpr[i], _ = metrics.roc_curve(
                        y_labels[:, i], y_probs[:, i], pos_label=1
                    )
                    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], label=f"class {i}")
            else:
                fpr, tpr, _ = metrics.roc_curve(y_labels, y_probs[:, 1], pos_label=1)
                plt.plot(fpr, tpr, label="positive class")

            fname = f"{plots_dir}/{key}.html"
            return context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)






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
