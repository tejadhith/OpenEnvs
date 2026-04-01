"""
Subprocess-isolated shader rendering harness.

Runs each render in a separate process to isolate from driver crashes,
infinite loops, and GPU state corruption.

Usage:
    from envs.shader.harness import render

    result = render("void mainImage(out vec4 c, in vec2 f) { c = vec4(1,0,0,1); }")
    print(result.compiled, result.rendered)
"""

import json
import subprocess
import sys
from base64 import b64decode
from dataclasses import dataclass, field
from pathlib import Path

RENDER_SCRIPT = str(Path(__file__).parent / "render.py")


@dataclass
class RenderResult:
    """Result of a shader render attempt."""

    compiled: bool = False
    rendered: bool = False
    errors: list[str] = field(default_factory=list)
    frame: bytes | None = None
    width: int = 0
    height: int = 0
    timed_out: bool = False


def render(code, resolution=(1280, 720), time=0.0, frame=0,
           timeout=10.0, mouse=(0.0, 0.0, 0.0, 0.0)):
    """
    Render a Shadertoy GLSL shader in an isolated subprocess.

    Args:
        code: Shadertoy-dialect GLSL fragment shader source
        resolution: (width, height) in pixels
        time: iTime value in seconds
        frame: iFrame value
        timeout: max seconds to wait for the subprocess
        mouse: (x, y, click_x, click_y) for iMouse

    Returns:
        RenderResult with compilation/render status, errors, and pixel data
    """
    request = json.dumps({
        "code": code,
        "resolution": list(resolution),
        "time": time,
        "frame": frame,
        "mouse": list(mouse),
    })

    try:
        proc = subprocess.run(
            [sys.executable, RENDER_SCRIPT],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return RenderResult(
            timed_out=True,
            errors=["subprocess timed out"],
            width=resolution[0], height=resolution[1],
        )

    if proc.returncode != 0:
        stderr = proc.stderr.strip() if proc.stderr else "unknown error"
        return RenderResult(
            errors=[f"subprocess crashed (exit {proc.returncode}): {stderr}"],
            width=resolution[0], height=resolution[1],
        )

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return RenderResult(
            errors=["failed to parse subprocess output"],
            width=resolution[0], height=resolution[1],
        )

    frame_bytes = None
    if data.get("frame"):
        frame_bytes = b64decode(data["frame"])

    return RenderResult(
        compiled=data.get("compiled", False),
        rendered=data.get("rendered", False),
        errors=data.get("errors", []),
        frame=frame_bytes,
        width=data.get("width", resolution[0]),
        height=data.get("height", resolution[1]),
    )
