from abc import ABC, abstractmethod
from enum import Enum

from mlrun.artifacts import Artifact


class ProductionStages(Enum):
    PRE_FIT = "pre_fit"
    POST_FIT = "post_fit"
    PRE_EVALUATION = "pre_evaluation"
    POST_EVALUATION = "post_evaluation"


class ArtifactPlan(ABC):
    """
    An abstract class for describing an artifact plan. A plan is used to produce artifact in a given time of a function
    according to the user's configuration.
    """

    @abstractmethod
    def validate(self, *args, **kwargs):
        """
        Validate this plan has the required data to produce its artifact.

        :raise ValueError: In case this plan is missing information in order to produce its artifact.
        """
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        """
        Check whether or not the plan fits the given stage for production.

        :param stage: The current stage in the function's process.

        :return: True if the given stage fits the plan and False otherwise.
        """

        pass

    @abstractmethod
    def produce(self, *args, **kwargs) -> Artifact:
        """
        Produce the artifact according to this plan.

        :return: The produced artifact.
        """