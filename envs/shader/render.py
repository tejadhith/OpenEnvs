"""
Headless GLSL shader renderer using ModernGL.

Renders Shadertoy-dialect GLSL fragment shaders to pixel buffers
via an EGL-backed OpenGL context (no display required).

Can be imported directly or run as a subprocess (used by harness.py):
    echo '{"code":"void mainImage(out vec4 c,in vec2 f){c=vec4(1,0,0,1);}"}' | python render.py
"""

import json
import platform
import re
import sys
from base64 import b64encode

VERTEX = """\
#version 330
in vec2 vert;
void main() {
    gl_Position = vec4(vert, 0.0, 1.0);
}
"""

# Shadertoy-compatible preamble. Declares all standard uniforms,
# forward-declares mainImage, and defines main() which calls it.
# User code (containing the mainImage definition) is appended after this.
PREAMBLE = """\
#version 330

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform float iFrameRate;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform vec3 iChannelResolution[4];
uniform float iChannelTime[4];
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

#define HW_PERFORMANCE 1

void mainImage(out vec4, in vec2);

out vec4 _fragcolor;

void main() {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(color, gl_FragCoord.xy);
    _fragcolor = color;
}

"""

PREAMBLE_LINES = PREAMBLE.count("\n")

# Fullscreen quad: 2 triangles covering [-1,1] x [-1,1].
QUAD = [
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
    -1.0,  1.0,
]


def preprocess(code):
    """Strip #version, precision, and #extension directives from user shader code."""
    code = re.sub(r"^\s*#version\s+.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*precision\s+\w+\s+\w+\s*;", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*#extension\s+.*$", "", code, flags=re.MULTILINE)
    return code


def _set(prog, name, value):
    """Set a uniform only if it exists in the compiled program."""
    if prog.get(name, None) is not None:
        prog[name] = value


def _adjust_errors(msg):
    """Extract error lines from ModernGL output and adjust line numbers."""
    lines = msg.strip().splitlines()
    errors = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if line in ("GLSL Compiler failed", "vertex_shader", "fragment_shader"):
            continue
        # Mesa format: "0:LINE(COL): error: ..."
        line = re.sub(
            r"^(\d+):(\d+)",
            lambda m: f"{m.group(1)}:{max(1, int(m.group(2)) - PREAMBLE_LINES)}",
            line,
        )
        # NVIDIA format: "0(LINE) : error ..."
        line = re.sub(
            r"^(\d+)\((\d+)\)",
            lambda m: f"{m.group(1)}({max(1, int(m.group(2)) - PREAMBLE_LINES)})",
            line,
        )
        errors.append(line)
    return errors if errors else [msg.strip()]


def _release(*objs):
    for obj in objs:
        try:
            obj.release()
        except Exception:
            pass


def render(code, resolution=(1280, 720), time=0.0, frame=0,
           mouse=(0.0, 0.0, 0.0, 0.0)):
    """
    Render a Shadertoy-dialect GLSL shader headlessly.

    Returns a dict:
        compiled:  bool
        rendered:  bool
        errors:    list[str]  (line numbers adjusted to user code)
        frame:     bytes | None  (raw RGBA, width*height*4, top-left origin)
        width:     int
        height:    int
    """
    try:
        import moderngl
        import numpy as np
    except ImportError as e:
        return {"compiled": False, "rendered": False,
                "errors": [f"missing dependency: {e}"],
                "frame": None, "width": 0, "height": 0}

    w, h = resolution
    fail = {"compiled": False, "rendered": False, "errors": [],
            "frame": None, "width": w, "height": h}

    fragment = PREAMBLE + preprocess(code)

    # Context
    try:
        backend = "egl" if platform.system() == "Linux" else None
        kwargs = {"backend": backend} if backend else {}
        ctx = moderngl.create_context(standalone=True, require=330, **kwargs)
    except Exception as e:
        fail["errors"] = [f"context: {e}"]
        return fail

    # Compile
    try:
        prog = ctx.program(vertex_shader=VERTEX, fragment_shader=fragment)
    except Exception as e:
        fail["errors"] = _adjust_errors(str(e))
        ctx.release()
        return fail

    # Setup framebuffer + geometry
    fbo = ctx.simple_framebuffer((w, h), components=4)
    fbo.use()

    vbo = ctx.buffer(np.array(QUAD, dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "vert")])

    # Uniforms
    _set(prog, "iResolution", (float(w), float(h), 1.0))
    _set(prog, "iTime", float(time))
    _set(prog, "iTimeDelta", 1.0 / 60.0)
    _set(prog, "iFrame", int(frame))
    _set(prog, "iFrameRate", 60.0)
    _set(prog, "iMouse", tuple(float(v) for v in mouse))
    _set(prog, "iDate", (2026.0, 1.0, 1.0, 0.0))
    _set(prog, "iSampleRate", 44100.0)

    # Render
    try:
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLES)
    except Exception as e:
        _release(vao, vbo, fbo, prog, ctx)
        fail["compiled"] = True
        fail["errors"] = [f"render: {e}"]
        return fail

    # Read pixels
    try:
        raw = fbo.read(components=4)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        pixels = np.flipud(pixels)
        frame_bytes = pixels.tobytes()
    except Exception as e:
        _release(vao, vbo, fbo, prog, ctx)
        fail["compiled"] = True
        fail["errors"] = [f"readback: {e}"]
        return fail

    _release(vao, vbo, fbo, prog, ctx)

    return {"compiled": True, "rendered": True, "errors": [],
            "frame": frame_bytes, "width": w, "height": h}


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.read())
        code = request.pop("code")
        result = render(code, **request)
        if result["frame"] is not None:
            result["frame"] = b64encode(result["frame"]).decode("ascii")
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"compiled": False, "rendered": False,
                   "errors": [str(e)], "frame": None,
                   "width": 0, "height": 0}, sys.stdout)
