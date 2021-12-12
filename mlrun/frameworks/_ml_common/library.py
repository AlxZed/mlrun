from abc import ABC, abstractmethod
from typing import List
from .._ml_common.plan import ArtifactPlan, ProductionStages
import scikitplot as skplt

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

            if hasattr(model, "predict_proba"):
                y_probas = model.predict_proba(apply_args['X_test'])
                skplt.metrics.plot_roc_curve(apply_args['y_test'], y_probas)
            else:
                print('wrong model')




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
