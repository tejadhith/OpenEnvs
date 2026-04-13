"""
Inference Script — Shader Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=gradient env=shader model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=<glsl_code> reward=0.45 done=false error=null
    [STEP] step=2 action=<glsl_code> reward=0.92 done=true error=null
    [END] success=true steps=2 score=0.92 rewards=0.45,0.92
"""

import asyncio
import os
import textwrap
from typing import List, Optional

try:
    from shader import ShaderAction, ShaderEnv
except ImportError:
    from envs.shader import ShaderAction, ShaderEnv
from openai import OpenAI

IMAGE_NAME = os.getenv("IMAGE_NAME")
if not IMAGE_NAME:
    raise RuntimeError(
        "IMAGE_NAME environment variable is required. "
        "Set it to the Docker image name (e.g. shader:latest)."
    )
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "shader"
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 4096
SUCCESS_THRESHOLD = 0.90

TASKS = ["Nd33R4", "stlXWH", "ftjSRd"]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a GLSL shader expert. Your task is to write a Shadertoy-dialect \
    GLSL fragment shader that reproduces the given reference image as closely \
    as possible.

    Rules:
    - Write a `void mainImage(out vec4 fragColor, in vec2 fragCoord)` function.
    - You may use standard Shadertoy uniforms: iResolution, iTime, iTimeDelta, \
    iFrame, iMouse, iDate, iSampleRate.
    - Do NOT include #version, precision, or #extension directives.
    - Output ONLY the raw GLSL code — no markdown fencing, no explanation.

    The rendered output is compared to the reference via SSIM (structural \
    similarity). Target: SSIM >= 0.99.""")


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action.replace("\n", " ")[:80]
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1
        while end > 0 and lines[end].strip() != "```":
            end -= 1
        if end > 0:
            return "\n".join(lines[1:end])
        return "\n".join(lines[1:])
    return text


FALLBACK_CODE = ("void mainImage(out vec4 fragColor, in vec2 fragCoord) "
                 "{ fragColor = vec4(0.0); }")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def call_model(client: OpenAI, conversation: list) -> str:
    """Send the conversation to the model and return stripped GLSL code."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
            + conversation,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return strip_fences(text) if text else ""
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

async def run_task(env, client: OpenAI, task_name: str) -> None:
    """Run one episode for a named task."""
    rewards: List[float] = []
    steps_taken = 0
    best_ssim = 0.0
    success = False
    conversation: list = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        ref_b64 = obs.reference_png

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # First step: send reference image; later steps: send feedback
            if step == 1:
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Write a GLSL shader that reproduces "
                                    "this reference image exactly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{ref_b64}",
                            },
                        },
                    ],
                })
            else:
                feedback = []
                if not obs.compiled:
                    feedback.append(
                        "Compilation FAILED.\nErrors:\n"
                        + "\n".join(obs.errors))
                elif not obs.rendered:
                    feedback.append(
                        "Render FAILED.\nErrors:\n"
                        + "\n".join(obs.errors))
                else:
                    feedback.append(f"SSIM: {obs.ssim:.4f} (need >= 0.99).")
                    feedback.append(
                        "Below is your current render vs the reference. "
                        "Fix the differences. Output ONLY raw GLSL code.")

                content: list = [
                    {"type": "text", "text": "\n".join(feedback)}
                ]
                if obs.agent_png:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{obs.agent_png}",
                        },
                    })
                conversation.append({"role": "user", "content": content})

            code = call_model(client, conversation) or FALLBACK_CODE

            # Step the environment
            result = await env.step(ShaderAction(code=code))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = "; ".join(obs.errors) if obs.errors else None

            rewards.append(reward)
            steps_taken = step
            best_ssim = max(best_ssim, obs.ssim)

            log_step(
                step=step, action=code, reward=reward,
                done=done, error=error,
            )

            conversation.append({"role": "assistant", "content": code})

            if done:
                break

        score = min(max(best_ssim, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(
            success=success, steps=steps_taken,
            score=score, rewards=rewards,
        )


# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await ShaderEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_name in TASKS:
            await run_task(env, client, task_name)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
