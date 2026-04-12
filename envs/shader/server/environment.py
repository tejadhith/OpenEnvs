"""
Shader environment implementation.

Reference-image-conditioned shader generation with SSIM reward.
"""

import random
from base64 import b64encode
from io import BytesIO
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from PIL import Image

try:
    from ..models import ShaderAction, ShaderObservation
    from ..tasks import Task, load as load_tasks, CURATED, CURATED_BY_NAME
    from ..reward import ssim
    from ..harness import render as harness_render
except ImportError:
    from models import ShaderAction, ShaderObservation
    from tasks import Task, load as load_tasks, CURATED, CURATED_BY_NAME
    from reward import ssim
    from harness import render as harness_render


class ShaderEnvironment(Environment):
    """
    OpenEnv environment for shader generation against a reference image.

    Each episode picks a task (shader with known ground-truth code), renders
    the reference frame, then challenges the agent to reproduce it via GLSL.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _tasks: list[Task] | None = None

    def __init__(self, budget: int = 10, seed: int | None = None):
        self._budget = budget
        self._rng = random.Random(seed)
        if ShaderEnvironment._tasks is None:
            ShaderEnvironment._tasks = load_tasks()
        self._state = State(episode_id=None, step_count=0)

        # Episode state
        self._task: Task | None = None
        self._ref: bytes | None = None
        self._remaining: int = 0

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              task: str | None = None, **kwargs) -> ShaderObservation:
        """Start a new episode.

        Args:
            seed: RNG seed for reproducible task selection.
            episode_id: Optional episode identifier.
            task: If given, select a curated task by name (e.g. "gradient",
                  "rings", "mandelbrot"). Otherwise picks randomly from the
                  full corpus.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4())[:8],
            step_count=0,
        )
        self._remaining = self._budget

        # Select task(s) to try
        if task and task in CURATED_BY_NAME:
            candidates = [CURATED_BY_NAME[task]]
        else:
            indices = list(range(len(self._tasks)))
            self._rng.shuffle(indices)
            candidates = [self._tasks[i] for i in indices]

        for t in candidates:
            result = harness_render(
                t.code,
                resolution=t.resolution,
                time=t.time,
            )
            if result.compiled and result.rendered and result.frame:
                self._task = t
                self._ref = result.frame
                return ShaderObservation(
                    task=t.name,
                    remaining=self._remaining,
                    reference_png=self._encode(
                        result.frame, result.width, result.height,
                    ),
                    done=False,
                    reward=None,
                )

        raise RuntimeError("no task rendered successfully")

    def step(self, action: ShaderAction) -> ShaderObservation:
        """Render agent's GLSL, compute SSIM vs reference, return observation."""
        if self._task is None:
            raise RuntimeError("call reset() before step()")

        self._state.step_count += 1
        self._remaining -= 1

        result = harness_render(
            action.code,
            resolution=self._task.resolution,
            time=self._task.time,
        )

        # Compute reward in [0.0, 1.0]
        if result.compiled and result.rendered and result.frame:
            score = ssim(
                self._ref, result.frame,
                result.width, result.height,
            )
            png = self._encode(
                result.frame, result.width, result.height,
            )
        else:
            score = 0.0
            png = ""

        done = self._remaining <= 0 or score >= 0.99

        return ShaderObservation(
            task=self._task.name,
            remaining=self._remaining,
            reference_png="",
            compiled=result.compiled,
            rendered=result.rendered,
            errors=result.errors,
            agent_png=png,
            ssim=score,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def _encode(rgba: bytes, width: int, height: int) -> str:
        """Convert raw RGBA bytes to a base64-encoded PNG string."""
        img = Image.frombytes("RGBA", (width, height), rgba)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return b64encode(buf.getvalue()).decode("ascii")
