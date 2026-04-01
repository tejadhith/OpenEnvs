# Shader Environment

## Overview

`shader` is an OpenEnv-compatible environment for generating, repairing, and iteratively refining executable shaders against visual and systems constraints.

Supported interaction modes:

- Prompt-conditioned or reference-conditioned shader creation
- Multi-turn shader refinement (not one-shot only)
- GLSL-first execution via headless rendering, with WGSL/browser-native as a later target
- Pluggable reward traces for RL, evaluation, reranking, and distillation
- Hidden evaluation across time, resolution, and seed variations

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
- **ModernGL** with EGL as the headless rendering backend
  - Runs on Linux servers without a display via EGL
  - Used by shaders21k for offline rendering
  - Each shader runs in an isolated subprocess with a timeout to contain driver crashes and infinite loops
- **`shaderc` / `glslang`** for offline validation and portability checks
- **WGSL / WebGPU** deferred to a later phase for browser-native demos

## Episode Schema

The environment follows a standard multi-turn refinement loop:

1. Start from an empty, partial, or broken shader
2. Provide a prompt, reference frame, reference clip, or target effect description
3. The agent edits the shader over multiple turns
4. Compile and render the shader after each turn
5. Render hidden evaluation frames across multiple conditions
6. Return structured reward traces after every turn

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
    shader_code: str


class ShaderObservation(Observation):
    brief: str
    shader_language: str
    visible_renders: dict[str, bytes]
    compiler_errors: list[str]
    perf_summary: dict
    robustness_summary: dict
    reward_breakdown: dict[str, float]
    diff_summary: str
    done: bool = False
    reward: float | None = None
```

- `reward` and `done` inside `Observation` follows OpenEnv spec (RFC 002, Decision 2): rewards are computed inside the environment and returned as part of the observation; the server layer promotes them to the top-level `StepResponse`.
- `reward_breakdown` aligns with OpenEnv's RFC 004 Rubric system for per-component reward logging.
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

The shaders21k corpus (21K OpenGL fragment shaders from Shadertoy) is the natural source for populating the initial task bank. RL training requires 10K-15K rollouts minimum based on adjacent work (CTRL-S: 14.4K, Reason-SVG: 10K).

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

## Next Steps

- Define the exact `reset()` and `step()` episode schema per OpenEnv RFC 002
- Build the first 20-50 task templates seeded from shaders21k
- Plan the SFT warm-up dataset and training run (prerequisite for RL)
- Implement the reward plugin API
- Establish the hidden evaluation protocol for appearance (DINOv2 + SSIM), stability, and performance

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
