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
    difficulty: str = ""  # "easy", "medium", "hard"


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


# -----------------------------------------------------------------------
# Curated tasks — 3 shaders from shaders21k corpus, selected by
# code complexity (line count, concepts required) and verified to
# render with good visual output at time=0.
# -----------------------------------------------------------------------

# Easy: XOR color pattern (13 lines, no loops, no iTime)
# Requires: int casting, bitwise XOR/AND, float conversion
_EASY_CODE = (
    'void mainImage( out vec4 fragColor, in vec2 fragCoord )\n'
    '{\n'
    '    int xor = (int(fragCoord.x) ^ int(fragCoord.y));\n'
    '    float r = float((xor * 2) & 0xff) / 255.0;\n'
    '    float g = float((xor * 4) & 0xff) / 255.0;\n'
    '\tfloat b = float((xor * 8) & 0xff) / 255.0;\n'
    '    \n'
    '    vec3 col = vec3(r, g, b);\n'
    '\tfragColor = vec4(col, 1.0);\n'
    '}\t\n'
    '\n'
    '\n'
    '/* See also: the 88 chars version in the comments!! */'
)

# Medium: SDF distance field visualization (44 lines)
# Requires: signed distance functions, abs, smoothstep, exp, cos for
# distance-based coloring, centered coordinates
_MEDIUM_CODE = (
    '// The MIT License\n'
    '// Copyright \u00a9 2021 Rodol Phito\n'
    '// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n'
    '\n'
    '// Signed distance to a square minus a circle\n'
    '\n'
    '// List of some other 2D distances: https://www.shadertoy.com/playlist/MXdSRf\n'
    '//\n'
    '// and www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm\n'
    '\n'
    'float sdSquareMinusCircle( in vec2 p, in float b ) \n'
    '{\n'
    '    p = abs(p); p = (p.y>p.x) ? p.yx : p.xy;\n'
    '    if (p.y<b*min(p.x,1.0)) return length(p-vec2(1.,b));\n'
    '    return max(length(max(p-1.,0.0)) + min(p.x-1.,0.0),sqrt(b*b+1.)-length(p));\n'
    '}\n'
    '\n'
    'void mainImage( out vec4 fragColor, in vec2 fragCoord )\n'
    '{\n'
    '\tvec2 p = 1.5*(2.0*fragCoord.xy-iResolution.xy)/iResolution.y;\n'
    '    vec2 m = 1.5*(2.0*iMouse.xy-iResolution.xy)/iResolution.y;\n'
    '\n'
    '    \n'
    '    // corner radius\n'
    '    float ra = 0.5*sin(iTime*1.2) + .5;\n'
    '\n'
    '    \n'
    '\tfloat d = sdSquareMinusCircle( p, ra );\n'
    '\n'
    '    \n'
    '    vec3 col = vec3(1.0) - sign(d)*vec3(0.1,0.4,0.7);\n'
    '\tcol *= 1.0 - exp(-3.0*abs(d));\n'
    '\tcol *= 0.8 + 0.2*cos(150.0*d);\n'
    '\tcol = mix( col, vec3(1.0), 1.0-smoothstep(0.0,0.015,abs(d)) );\n'
    '\n'
    '    if( iMouse.z>0.001 )\n'
    '    {\n'
    '    d = sdSquareMinusCircle(m, ra );\n'
    '    col = mix(col, vec3(1.0,1.0,0.0), 1.0-smoothstep(0.0, 0.005, abs(length(p-m)-abs(d))-0.0025));\n'
    '    col = mix(col, vec3(1.0,1.0,0.0), 1.0-smoothstep(0.0, 0.005, length(p-m)-0.015));\n'
    '    }\n'
    '    \n'
    '\tfragColor = vec4(col,1.0);\n'
    '}'
)

# Hard: Raymarcher with SDF repetition, HSV coloring (122 lines)
# Requires: ray marching loop, rotation matrices, SDF primitives,
# domain repetition, HSV-to-RGB conversion, polar coordinates
_HARD_CODE = (
    '#define time iTime*0.6\n'
    'mat2 r2d(float a) {\n'
    '    return mat2(cos(a),sin(a),-sin(a),cos(a));\n'
    '}\n'
    '\n'
    '//from iqs sdf functions page\n'
    'vec3 opRepLim( in vec3 p, in float c, in vec3 l)\n'
    '{\n'
    '    vec3 q = p-c*clamp(round(p/c),-l,l);\n'
    '    return q;\n'
    '    //return primitive( q );\n'
    '}\n'
    '// All components are in the range [0\u20261], including hue.\n'
    'vec3 rgb2hsv(vec3 c)\n'
    '{\n'
    '    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);\n'
    '    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));\n'
    '    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));\n'
    '\n'
    '    float d = q.x - min(q.w, q.y);\n'
    '    float e = 1.0e-10;\n'
    '    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);\n'
    '}\n'
    ' \n'
    '\n'
    '// All components are in the range [0\u20261], including hue.\n'
    'vec3 hsv2rgb(vec3 c)\n'
    '{\n'
    '    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n'
    '    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n'
    '    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n'
    '}\n'
    '//Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl.\n'
    '\n'
    'vec4 map(vec3 p) {\n'
    '    float pd = 1.;\n'
    '    //p.z += time*2.1;\n'
    '    vec3 o = p;\n'
    '    float c = length(p);\n'
    '    //p.xy *= r2d(sin(time*1.)+time);\n'
    '    //p.xz *= r2d(sin(time*1.+c*13.));\n'
    '    p.xy *= r2d(sin(c*50.+time)*0.2);\n'
    '    //p = (fract(p*pd)-0.5)/pd;\n'
    '    //float ramnt = clamp((sin(time*1.)),0.,1.);\n'
    '    float rt = time*0.5;\n'
    '    float ramnt = sin((3.14159/2.)*cos(rt)+rt);\n'
    '    ramnt -= 1.;\n'
    '    p.xz *= r2d(sin(c*0.5+time*0.3)*4.*ramnt);\n'
    '    p.yz *= r2d(cos(c*0.5+time*0.3)*4.*ramnt);\n'
    '    p.xy = vec2(length(p.xy),atan(p.x,p.y));\n'
    '    //p.y = abs(p.y);\n'
    '    //p.y *= 8./3.14159;\n'
    '    p.y *= 1./3.1415;\n'
    '    p.y *= (sin(time*0.13-2.)*0.5+0.5)*10.;\n'
    '    p.y *= 0.5;\n'
    '    p.y += p.z*sin(time);\n'
    '    p.y = (fract(p.y)-0.5);\n'
    '    p.y = abs(p.y);\n'
    '    p.xy = vec2(p.x*sin(p.y),p.x*cos(p.y));\n'
    '    p = opRepLim(p,0.21,vec3(1.));\n'
    '    p.xz *= r2d(sin(time*1.+c*13.));\n'
    '    p = opRepLim(p,0.018,vec3(2.));\n'
    '    p.xy *= r2d(sin(time*1.)+time);\n'
    '    p = opRepLim(p,mix(0.01,0.003,(sin(time*0.35+15.)*0.5+0.5)),vec3((0.5-c)*10.));\n'
    '\n'
    '    float d = length(p)+0.005*sin(time*0.1+c*5.);\n'
    '    return vec4(p,d);\n'
    '}\n'
    '\n'
    '\n'
    '\n'
    'vec2 RM(vec3 ro, vec3 rd) {\n'
    '    float dO = 0.;\n'
    '    float ii = 0.;\n'
    '    \n'
    '    for (int i=0;i<200;i++) {\n'
    '        vec3 p = ro+rd*dO;\n'
    '        float dS = map(p).w;\n'
    '        dO += dS*0.5;\n'
    '        ii += 0.1;\n'
    '        if (dS < 0.001 || dO > 1000.) {break;}\n'
    '    }\n'
    '    return vec2(dO,ii);\n'
    '}\n'
    '\n'
    'void mainImage( out vec4 fragColor, in vec2 fragCoord )\n'
    '{\n'
    '    // Normalized pixel coordinates (from 0 to 1)\n'
    '    vec2 uv = -1. + 2.* fragCoord/iResolution.xy;\n'
    '\n'
    '    // Time varying pixel color\n'
    '    //vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n'
    '\n'
    '    //vec2 uv = -1. + 2. * inData.v_texcoord;\n'
    '    vec2 R = iResolution.xy;\n'
    '    float ar = R.x/R.y;\n'
    '    uv.x *= ar;\n'
    '    vec3 col = vec3(0.);\n'
    '    float c = length(uv*0.9);\n'
    '    //uv *= r2d(c*4.+time);\n'
    '    //uv *= r2d(uv.y*time*.1-time*4.3);\n'
    '    vec3 ro = vec3(0.,0.,-0.5);\n'
    '    //ro.z += time*0.2;\n'
    '    vec3 rd = normalize(vec3(uv,1.));\n'
    '    vec2 d = RM(ro,rd);\n'
    '    \n'
    '    //col = sin(uv.xyy*39.);\n'
    '    col = vec3((d.y*2.15)-0.3);\n'
    '    col = sin(d.yyy*0.1);\n'
    '    //col -= d.y*0.3;\n'
    '    vec3 hsv = vec3(\n'
    '    sin(sin(d.x)*40.)*0.3+time*0.1,\n'
    '    sin(d.x*300.)*0.5+0.5,\n'
    '    (d.y*0.1)\n'
    '    );\n'
    '    if (d.x > 100.) {\n'
    '        hsv.y *= 0.2;\n'
    '    }\n'
    '    col = hsv2rgb(hsv);\n'
    '    \n'
    '    fragColor = vec4(col,1.);\n'
    '}'
)

CURATED: list[Task] = [
    Task(
        name="Nd33R4",
        code=_EASY_CODE,
        source="https://www.shadertoy.com/view/Nd33R4",
        difficulty="easy",
    ),
    Task(
        name="stlXWH",
        code=_MEDIUM_CODE,
        source="https://www.shadertoy.com/view/stlXWH",
        difficulty="medium",
    ),
    Task(
        name="ftjSRd",
        code=_HARD_CODE,
        source="https://www.shadertoy.com/view/ftjSRd",
        difficulty="hard",
    ),
]

CURATED_BY_NAME: dict[str, Task] = {t.name: t for t in CURATED}


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
