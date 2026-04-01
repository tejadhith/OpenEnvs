"""Shader environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ShaderAction, ShaderObservation


class ShaderEnv(EnvClient[ShaderAction, ShaderObservation, State]):
    """
    Client for the shader environment.

    Example:
        >>> with ShaderEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     result = client.step(ShaderAction(code="void mainImage(...)"))
        ...     print(result.observation.ssim)
    """

    def _step_payload(self, action: ShaderAction) -> Dict:
        return {"code": action.code}

    def _parse_result(self, payload: Dict) -> StepResult[ShaderObservation]:
        obs_data = payload.get("observation", {})
        observation = ShaderObservation.model_validate({
            **obs_data,
            "done": payload.get("done", False),
            "reward": payload.get("reward"),
        })
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State.model_validate(payload)
