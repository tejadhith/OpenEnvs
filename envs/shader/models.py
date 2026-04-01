"""Data models for the shader environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ShaderAction(Action):
    """Agent submits Shadertoy-dialect GLSL fragment shader source."""

    code: str = Field(..., description="Shadertoy-dialect GLSL fragment shader source")


class ShaderObservation(Observation):
    """Observation returned after reset or step."""

    # Task context
    task: str = Field(default="")
    remaining: int = Field(default=0)

    # Reference (non-empty on reset only)
    reference_png: str = Field(
        default="",
        description="Base64-encoded PNG of the reference render. Non-empty on reset only.",
    )

    # Render result
    compiled: bool = Field(default=False)
    rendered: bool = Field(default=False)
    errors: list[str] = Field(default_factory=list)
    agent_png: str = Field(
        default="",
        description="Base64-encoded PNG of the agent's render. Empty if compile failed.",
    )

    # Reward signal
    ssim: float = Field(default=0.0, description="SSIM vs reference in [0, 1]")
