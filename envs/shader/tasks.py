"""
Task bank for the shader environment.

Loads Shadertoy-dialect GLSL shaders from the shaders21k dataset.
The environment renders the ground-truth to produce a reference image,
then challenges the agent to reproduce it.

Run `./download.sh` to fetch the dataset before first use.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "shadertoy"


@dataclass
class Task:
    name: str
    code: str
    source: str = ""
    resolution: tuple[int, int] = (512, 288)
    time: float = 0.0


def _parse(path: Path) -> Task | None:
    """Parse a single .fragment JSON file into a Task, or None if unusable."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    info = data.get("info", {})
    passes = data.get("renderpass", [])

    if len(passes) != 1:
        return None

    rp = passes[0]

    # Skip shaders that need texture/buffer inputs
    if rp.get("inputs"):
        return None

    code = rp.get("code", "")
    if not code or "mainImage" not in code:
        return None

    sid = info.get("id", path.stem)

    return Task(
        name=sid,
        code=code,
        source=f"https://www.shadertoy.com/view/{sid}",
    )


def load() -> list[Task]:
    """Load all usable single-pass shaders from the dataset."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Shader dataset not found at {DATA_DIR}. "
            "Run ./download.sh to fetch it."
        )

    tasks = []
    for fragment in sorted(DATA_DIR.rglob("*.fragment")):
        task = _parse(fragment)
        if task is not None:
            tasks.append(task)

    log.info("loaded %d tasks from %s", len(tasks), DATA_DIR)
    return tasks
