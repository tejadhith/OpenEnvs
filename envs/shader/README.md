---
title: Shader Environment Server
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Shader Environment

## Overview

`shader` is an OpenEnv-compatible environment for generating, repairing, and iteratively refining executable shaders against visual and systems constraints.

Supported interaction modes:

- Reference-conditioned shader recreation via SSIM reward
- Multi-turn shader refinement (not one-shot only)
- GLSL-first execution via headless rendering, with WGSL/browser-native as a later target
- Pluggable reward traces for RL, evaluation, reranking, and distillation
- Hidden evaluation across time, resolution, and seed variations

## Project Structure

```
envs/shader/
├── __init__.py          # Public API: ShaderEnv, ShaderAction, ShaderObservation
├── models.py            # Pydantic Action / Observation types
├── client.py            # OpenEnv client (ShaderEnv)
├── tasks.py             # Task bank loader (shaders21k corpus)
├── reward.py            # SSIM reward computation
├── render.py            # Headless GLSL renderer (ModernGL + EGL)
├── harness.py           # Subprocess-isolated render wrapper
├── download.sh          # Fetches shaders21k dataset
├── openenv.yaml         # OpenEnv space descriptor
├── pyproject.toml
└── server/
    ├── app.py           # FastAPI application (OpenEnv HTTP server)
    ├── environment.py   # Environment implementation (reset / step loop)
    └── Dockerfile
```

## Quickstart

```bash
# Download the shaders21k corpus (~41 MB)
cd envs/shader
./download.sh
```

### Running via uv / uvicorn

```bash
cd envs/shader
PYTHONPATH=../.. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running via Docker

The corpus is downloaded at build time automatically:

```bash
cd envs/shader
docker build -f server/Dockerfile -t shader .
docker run -p 8000:8000 shader
```

### Validation

```bash
# Validate local structure
cd envs/shader
openenv validate --verbose

# Validate a running server (6 criteria: openapi, health, metadata, schema, mcp, mode)
openenv validate http://localhost:8000
```

### Interacting via WebSocket

The HTTP endpoints (`/reset`, `/step`) are stateless — each creates a fresh environment instance. For multi-turn sessions with persistent state, use the WebSocket endpoint:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset — picks a task, renders reference
        await ws.send(json.dumps({"type": "reset", "data": {}}))
        resp = json.loads(await ws.recv())
        obs = resp["data"]["observation"]
        print(obs["task"], obs["remaining"])

        # Step — submit GLSL, get back SSIM + render
        await ws.send(json.dumps({
            "type": "step",
            "data": {"code": "void mainImage(out vec4 c, in vec2 f) { c = vec4(1,0,0,1); }"}
        }))
        resp = json.loads(await ws.recv())
        r = resp["data"]
        print(f"compiled={r['observation']['compiled']} ssim={r['observation']['ssim']} reward={r['reward']}")

        await ws.send(json.dumps({"type": "close"}))

asyncio.run(main())
```

### Python Client

```python
from shader import ShaderEnv, ShaderAction

with ShaderEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    print(result.observation.task)         # ShaderToy ID
    print(result.observation.reference_png) # base64 PNG

    result = client.step(ShaderAction(code="void mainImage(out vec4 c, in vec2 f) { c = vec4(1,0,0,1); }"))
    print(result.observation.compiled)     # True/False
    print(result.observation.ssim)         # similarity vs reference
```

### Benchmark

Runs GPT 5.4 against the environment over WebSocket, producing a reproducible baseline:

```bash
# Requires a running server and OPENAI_API_KEY set
python envs/shader/benchmark.py                                    # 3 episodes, default seeds
python envs/shader/benchmark.py --turns 5                          # cap turns per episode
python envs/shader/benchmark.py --url ws://localhost:8001/ws       # custom server
python envs/shader/benchmark.py --seeds 10 20 30                   # custom seeds
```

Seeds control reproducible task selection. Results are saved to `benchmark_output/results.json`.

## Tasks

The environment ships with 3 curated tasks at increasing difficulty. Each task presents a reference image; the agent must write GLSL code to reproduce it. The grader is SSIM (structural similarity), returning a score in [0.0, 1.0].

| Task | Difficulty | Lines | Description | What the agent needs |
|------|-----------|-------|-------------|---------------------|
| `Nd33R4` | Easy | 13 | XOR color pattern on pixel coordinates | `int()` casting, bitwise XOR/AND, float conversion |
| `stlXWH` | Medium | 44 | SDF distance field (square minus circle) with smooth coloring | Signed distance functions, `abs`, `exp`, `smoothstep`, `cos` for distance coloring |
| `ftjSRd` | Hard | 122 | Raymarcher with SDF repetition, polar coordinates, HSV coloring | Ray marching loop, rotation matrices, domain repetition, HSV-to-RGB, polar coords |

All 3 tasks are sourced from the [shaders21k](https://github.com/mbaradad/shaders21k) corpus (Shadertoy). They were selected by code complexity (line count, number of concepts) and verified to produce visually interesting output at render time.

To select a specific task, pass its name to `reset()`:

```python
result = env.reset(task="Nd33R4")   # easy — XOR pattern
result = env.reset(task="stlXWH")   # medium — SDF visualization
result = env.reset(task="ftjSRd")   # hard — raymarcher
result = env.reset()                # random from full corpus
```

### Grading

Each task uses the same grader: SSIM between the agent's rendered output and the ground-truth reference image. The score is deterministic and reproducible for the same GLSL input.

- **Score range**: 0.0 (no similarity) to 1.0 (pixel-perfect match)
- **Success threshold**: score >= 0.90
- **Compile/render failures**: score = 0.0

### Baseline Scores

Evaluated with GPT 5.4 via `inference.py` (5 steps per task, temperature 0.2):

| Task | Difficulty | Best SSIM | Step-by-step rewards | Multi-turn improvement |
|------|-----------|-----------|---------------------|----------------------|
| `Nd33R4` | Easy | **0.27** | 0.27, 0.13, 0.15, 0.15, 0.16 | No — model generates hash noise instead of XOR pattern |
| `stlXWH` | Medium | **0.94** | 0.85, 0.88, 0.91, 0.90, 0.94 | Yes — steady refinement from 0.85 to 0.94 |
| `ftjSRd` | Hard | **0.40** | 0.31, 0.15, 0.33, 0.40, 0.28 | Partial — oscillates between 0.15-0.40 |

Key observations:
- `Nd33R4` (easy by code complexity) is hard for VLMs because bitwise XOR patterns are difficult to reverse-engineer from a rendered image alone. The model defaults to procedural noise rather than integer math.
- `stlXWH` (medium) shows clear multi-turn refinement — the model progressively improves the SDF shape and color mapping across steps.
- `ftjSRd` (hard) challenges frontier models with 122 lines of tightly coupled raymarching, rotation, and HSV coloring. The model attempts structural elements but cannot match the exact parameters.

*Run `inference.py` to reproduce. Scores vary by model and API endpoint.*

## Task Bank (Corpus)

Beyond the 3 curated tasks, the full task bank is loaded from the [shaders21k](https://github.com/mbaradad/shaders21k) corpus (NeurIPS 2022). At load time, shaders are filtered to single-pass fragments with no texture/buffer inputs, yielding ~16,800 usable tasks.

Each task is a Shadertoy-dialect GLSL shader with known ground-truth code. The environment renders the ground truth to produce a reference image, then challenges the agent to reproduce it.

| Field | Description |
|-------|-------------|
| `name` | ShaderToy ID (e.g. `MdGcDc`) |
| `code` | GLSL fragment shader source |
| `source` | ShaderToy URL for provenance |
| `resolution` | Render resolution, default 512x288 |
| `time` | iTime uniform value, default 0.0 |
| `difficulty` | `easy`, `medium`, `hard` (curated tasks only) |

## Motivation

Shader work has properties that make it well-suited as an RL environment:

- Feedback is fast and dense
- Compile success, render success, and performance are easy to gate
- The same shader can be evaluated under multiple controlled render conditions
- An active public corpus exists (Shadertoy, ~1M public shaders) alongside established compiler infrastructure (`shaderc`, `glslang`)

ShaderEval (LLM4Code @ ICSE 2025) benchmarked current LLMs on GLSL function completion and found failure rates above 31% even for top models. GLSL is low-resource in pretraining corpora, leaving room for RL-trained or fine-tuned models to improve on baselines.

### Positioning

There is existing adjacent work on conditioned procedural material generation, RL-based material parameter optimization, interactive evolutionary shader generation, and LLM-driven real-time shader generation. The contribution here is not "LLMs can emit shader code" but rather a reusable environment layer with stable runtime, task packaging, and reward/eval plugins.

## Runtime Stack

The primary target is GLSL via headless OpenGL.

- **GLSL** as the authoring and execution language
  - The entire relevant corpus is GLSL: Shadertoy (~1M shaders), shaders21k (21K shaders), ShaderEval (the only published GLSL benchmark)
  - GLSL has explicit representation in LLM pretraining corpora (The Stack includes a GLSL subset); WGSL has near-zero public training data
  - WGSL prohibits recursion, has no implicit type coercions, and uses incompatible uniform conventions, making Shadertoy-dialect GLSL non-transpilable to WGSL via naga at corpus scale (~15% failure rate)
- **ModernGL** with EGL as the headless rendering backend (`render.py`)
  - Runs on Linux servers without a display via EGL
  - Used by shaders21k for offline rendering
  - Wraps user shader code in a Shadertoy-compatible preamble (standard uniforms, `mainImage` forward declaration)
  - Strips `#version`, `precision`, and `#extension` directives from user code to avoid conflicts
  - Adjusts error line numbers reported by the driver to map back to user code
- **Subprocess isolation** (`harness.py`)
  - Each render runs in a separate process to contain driver crashes, infinite loops, and GPU state corruption
  - Configurable per-render timeout (default 10s)
  - Returns structured `RenderResult` with compile/render status, error messages, and raw RGBA pixel data
- **`shaderc` / `glslang`** for offline validation and portability checks (planned)
- **WGSL / WebGPU** deferred to a later phase for browser-native demos

## Episode Schema

The environment follows a standard multi-turn refinement loop:

1. `reset()` picks a task from the bank, renders the ground-truth reference, and returns it as a base64 PNG
2. The agent submits GLSL code via `step(ShaderAction(code=...))`
3. The server compiles and renders the shader, computes SSIM vs reference
4. Returns compile/render status, errors, rendered image, and SSIM reward
5. Episode ends when the turn budget is exhausted (default 10 turns) or SSIM >= 0.99

The server supports up to 4 concurrent environment sessions (`max_concurrent_envs=4`).

This supports RL training, best-of-N search and reranking, trajectory collection for SFT or distillation, and evaluation under a shared runtime.

## Environment Variants

### `shader` (primary)

Match a target still image or short animated effect within a frame-time budget. Direct, visual, and evaluable without a full DCC toolchain.

### Material Graph

Edit procedural material graphs and parameters toward a target appearance. The search space is more structured than raw shader code, and recent procedural-material work already uses graph/program representations.

### Shader Repair

Start from shaders that are broken, slow, unstable, or portability-problematic and optimize for correctness, robustness, and performance.

## Task Families

- **Reference recreation** — recreate a target still or short effect from a reference render
- **Repair** — fix syntax errors, portability failures, or numerical instability
- **Optimization** — preserve appearance while reducing frame time or instruction count
- **Style transfer** — preserve scene logic while shifting color, texture, motion, or lighting style
- **Critique incorporation** — revise the shader based on iterative feedback
- **Robustness repair** — stabilize a shader across resolutions, aspect ratios, and time ranges

## Evaluation

Evaluation relies on hidden checks rather than visible examples only:

- Compile success
- Render success (no NaNs or fatal runtime failures)
- Perceptual similarity on held-out stills
- Temporal consistency on held-out short clips
- Stability across resolutions, aspect ratios, and parameter seeds
- Frame-time or instruction-budget limits
- Optional portability checks across compiler/validator paths

### Reward Structure

**Current implementation** (`reward.py`): windowed SSIM (Wang et al. 2004) between agent render and reference, computed per-channel on RGB and averaged. Uses scipy `uniform_filter` for windowed statistics when available, falls back to global-stats SSIM. Compile and render failures receive reward 0.0. Reward range: [0.0, 1.0].

**Planned multi-component reward:**

```text
R = G_compile * G_render * (
  0.35 * appearance_match +
  0.20 * temporal_stability +
  0.20 * performance +
  0.15 * robustness +
  0.10 * code_quality
) - step_penalty - regression_penalty
```

Component notes:

- **`appearance_match`** — measured on hidden render conditions using DINOv2 cosine similarity (better than CLIP for texture/color/style fidelity) combined with pixel-level SSIM for structural accuracy. CLIP is appropriate only for text-conditioned task variants.
- **`temporal_stability`** — requires rendering N consecutive frames and computing frame-to-frame SSIM. For v1, this may serve as a held-out evaluation metric rather than a dense training reward to keep per-episode compute manageable.
- **`performance`** — frame-time as a reward is tractable headlessly. To avoid reward hacking (trivially simple shaders that are fast but visually wrong), this is gated as a hard budget first (penalize frames over a threshold) before adding a continuous score.
- **`robustness`** — captures resolution changes, seed changes, and compiler portability.
- **`G_compile * G_render`** — hard multiplicative gates, standard in code generation RL. These cause a zero-gradient problem early in training; mitigate with curriculum learning (start from partial or working shader skeletons) and/or soft penalties before hardening.
- **SFT warm-up** is a prerequisite before RL. Adjacent work (RLRF for SVG) shows that RL directly on an instruction-tuned model without a domain SFT stage fails because the base model cannot generate renderable output reliably enough to produce reward variance.

## Action / Observation Contract

```python
class ShaderAction(Action):
    code: str  # Shadertoy-dialect GLSL fragment shader source

class ShaderObservation(Observation):
    task: str               # ShaderToy ID
    remaining: int          # turns left in episode
    reference_png: str      # base64 PNG (non-empty on reset only)
    compiled: bool
    rendered: bool
    errors: list[str]
    agent_png: str          # base64 PNG of agent's render
    ssim: float             # SSIM vs reference in [0, 1]
    done: bool = False
    reward: float | None = None
```

- `reward` and `done` inside `Observation` follows OpenEnv spec (RFC 002, Decision 2): rewards are computed inside the environment and returned as part of the observation; the server layer promotes them to the top-level `StepResponse`.
- Detailed artifacts (frame dumps, profiler traces, held-out evaluation results) live behind tools rather than being inlined on every turn.

## Training

### Algorithm

- **GRPO** as the baseline, with multi-turn extensions:
  - **MURPHY** (NeurIPS 2025) — for each rollout that does not reach maximum reward, execution feedback is appended and new rollouts are generated from that state. Up to 8% relative gain over single-turn GRPO.
  - **TRLOO** (Dr. Kernel) — addresses the biased policy gradient problem that vanilla GRPO has in multi-turn settings.
- The episode schema (compile, render, reward, next turn) maps directly onto MURPHY's interaction loop.

### Libraries

- **veRL**: supports GRPO, DAPO, GSPO; scales to large models; best throughput for long multi-turn rollouts
- **OpenRLHF-M**: multimodal variant of OpenRLHF; targets VLM policies (code + rendered image inputs)

Both have TRL and OpenEnv integrations.

### Task Bank

The task bank is populated from the shaders21k corpus (~16.8K single-pass shaders after filtering). RL training requires 10K-15K rollouts minimum based on adjacent work (CTRL-S: 14.4K, Reason-SVG: 10K).

## Related Work

| Work | Relevance |
|------|-----------|
| **Shadertoy** | Active public corpus and community; seed source for tasks and reference effects |
| **OpenEnv** (Meta/PyTorch) | Target framework. Client-server RL environment with `Action`/`Observation` base classes, `reset()`/`step()` contract, and RFC 004 Rubric system. TRL, Unsloth, SkyRL, and Oumi integrations. |
| **ShaderEval** (LLM4Code @ ICSE 2025) | Only formal benchmark for GLSL code generation evaluation. 467 functions, pixel-diff evaluation via `shadermatch`. |
| **shaders21k** (NeurIPS 2022) | 21K OpenGL fragment shaders from Shadertoy for visual representation learning. Ready seed corpus for task generation. |
| **AI Co-Artist** (arXiv:2512.08951) | Closest work to multi-turn LLM-driven GLSL refinement. GPT-4 + Picbreeder-style evolution, <3% compile error after retries. No RL. |
| **VLMaterial** (ICLR 2025 Spotlight) | Fine-tunes a VLM for Blender material node graphs from images. Validates rendered-image similarity as a reward signal for code-generating policies. |
| **Dr. Kernel / KernelGYM** (arXiv:2602.05885) | Multi-turn RL for GPU code generation (CUDA/Triton) with compile-correctness-speedup reward chain. Proposes TRLOO for multi-turn credit assignment. |
| **MURPHY** (NeurIPS 2025) | Multi-turn GRPO for code generation. Canonical algorithm for the interaction loop used here. |
| **ProcMatRL** (SIGGRAPH Asia 2024) | RL for procedural material parameter optimization. Validates RL applicability in the visual generation domain. |
| **ShadAR** (arXiv:2602.17481) | LLM-driven real-time shader generation for AR. |
| **Procedural Shader Evolution** (arXiv:2312.17587) | Interactive evolutionary algorithms for shader generation. Multi-turn refinement as iterative optimization. |

## Licensing

Shader code in the task bank is sourced from [ShaderToy](https://www.shadertoy.com/) via the [shaders21k](https://github.com/mbaradad/shaders21k) dataset. ShaderToy's default license is **CC BY-NC-SA 3.0** — authors may choose a different license, but there is no structured per-shader license metadata in the dataset or the ShaderToy API.

- The dataset is not redistributed in this repository; it is downloaded at build time
- Individual shader provenance is tracked via the `source` field on each task (links back to the ShaderToy page)
- For commercial use, per-shader license review is required

## Next Steps

- Plan the SFT warm-up dataset and training run (prerequisite for RL)
- Extend reward beyond SSIM: DINOv2 appearance match, temporal stability, performance, robustness
- Implement the reward plugin API (multi-component, pluggable)
- Establish the hidden evaluation protocol across time, resolution, and seed variations
- Add `shaderc` / `glslang` offline validation
- Add support for multi-pass shaders and texture inputs

## References

### Primary

- Shadertoy: <https://www.shadertoy.com/>
- OpenEnv (Meta/PyTorch): <https://github.com/meta-pytorch/OpenEnv>
- OpenEnv RFC 002: <https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/002-env-spec.md>
- OpenEnv RFC 004: <https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md>
- TRL OpenEnv integration: <https://huggingface.co/docs/trl/openenv>
- ShaderEval (LLM4Code @ ICSE 2025): <https://conf.researchr.org/details/icse-2025/llm4code-2025-papers/13/Evaluating-Language-Models-for-Computer-Graphics-Code-Completion>
- shadertoys-dataset: <https://github.com/Vipitis/shadertoys-dataset>
- Shadereval-inputs (HuggingFace): <https://huggingface.co/datasets/Vipitis/Shadereval-inputs>
- shaders21k (NeurIPS 2022): <https://arxiv.org/abs/2211.16412>
- shaders21k dataset: <https://github.com/mbaradad/shaders21k>
- AI Co-Artist: <https://arxiv.org/abs/2512.08951>
- VLMaterial (ICLR 2025): <https://arxiv.org/abs/2501.18623>
- VLMaterial code: <https://github.com/mit-gfx/VLMaterial>
- Dr. Kernel / KernelGYM: <https://arxiv.org/abs/2602.05885>
- MURPHY (NeurIPS 2025): <https://arxiv.org/abs/2511.07833>
- RLRF (SVG RL): <https://arxiv.org/abs/2505.20793>

### Tools and Infrastructure

- `shaderc`: <https://github.com/google/shaderc>
- `glslang`: <https://github.com/KhronosGroup/glslang>
- `pygfx/shadertoy`: <https://github.com/pygfx/shadertoy>
- moderngl: <https://github.com/moderngl/moderngl>
- veRL: <https://github.com/volcengine/verl>
- OpenRLHF-M: <https://github.com/OpenRLHF/OpenRLHF-M>
- Adobe ProcMatRL: <https://github.com/adobe-research/ProcMatRL>

### Specifications

- WGSL specification: <https://www.w3.org/TR/WGSL/>
- WebGPU: <https://webgpu.org/>
- Khronos `glslang` reference: <https://www.khronos.org/opengles/sdk/Reference-Compiler/>

### Supplementary

- Procedural Shader Evolution: <https://arxiv.org/abs/2312.17587>
- ShadAR: <https://arxiv.org/abs/2602.17481>
- ProcMatRL paper (SIGGRAPH Asia 2024): <https://doi.org/10.1145/3687979>
- Conditioned Procedural Materials (SIGGRAPH 2023): <https://dl.acm.org/doi/10.1145/3588432.3591520>
- FragCoord.xyz: <https://fragcoord.xyz/>
- naga GLSL front-end failures: <https://github.com/Vipitis/shadertoys-dataset/issues/15>
