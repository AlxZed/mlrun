from typing import List, Dict

from mlrun.artifacts import Artifact
from plan import ArtifactPlan, ProductionStages

class ArtifactsPlansManager:
    def __init__(self, plans: List[ArtifactPlan]):
        self._plans = plans
        self._artifacts = []  # type: List[Artifact]

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        return {artifact.key: artifact for artifact in self._artifacts}

    def validate(self):
        #call validation of each plan
        pass

    def generate(self, model, context, apply_args, mystage: ProductionStages, *args, **kwargs):
        self._artifacts += [
            plan.produce(model, context, apply_args, **kwargs)
            for plan in self._plans
            if plan.is_ready(stage=mystage)
        ]

