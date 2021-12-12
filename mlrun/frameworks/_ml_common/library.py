import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from itertools import cycle
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

