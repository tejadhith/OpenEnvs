#!/usr/bin/env python3
"""
Benchmark GPT 5.4 against the shader environment via WebSocket.

Connects to a running shader environment server and runs a multi-turn
agent loop where GPT 5.4 tries to reproduce each reference image in GLSL.

Usage:
    # Start the server first:
    #   uvicorn server.app:app --host 0.0.0.0 --port 8000
    #   OR: docker run -p 8000:8000 shader

    python envs/shader/benchmark.py                          # run 3 episodes
    python envs/shader/benchmark.py --turns 5                # cap turns
    python envs/shader/benchmark.py --url ws://localhost:8001/ws  # custom server
"""

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import websockets
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_output"

# ---------------------------------------------------------------------------
# OpenAI client (Responses API)
# ---------------------------------------------------------------------------
_client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
if os.environ.get("OPENAI_BASE_URL"):
    _client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

CLIENT = OpenAI(**_client_kwargs)
MODEL = "gpt-5.4"

INSTRUCTIONS = """\
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
similarity). Target: SSIM >= 0.99."""


# ---------------------------------------------------------------------------
# Helpers
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


def extract_text(response) -> str:
    """Pull text from a Responses API response object."""
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    return block.text
    return ""


def save_b64_png(b64: str, path: Path):
    """Save a base64-encoded PNG string to a file."""
    path.write_bytes(base64.b64decode(b64))


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------

async def ws_send(ws, msg_type: str, data: dict) -> dict:
    """Send a message and return the response data."""
    await ws.send(json.dumps({"type": msg_type, "data": data}))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        raise RuntimeError(f"Server error: {resp.get('data', {})}")
    return resp["data"]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_episode(ws, seed: int, episode_dir: Path, max_turns: int) -> dict:
    """Run one episode via WebSocket. Returns result dict."""
    # Reset
    data = await ws_send(ws, "reset", {"seed": seed})
    obs = data["observation"]
    task = obs["task"]
    ref_b64 = obs["reference_png"]
    remaining = obs["remaining"]

    print(f"  task: {task}, budget: {remaining}")

    # Save reference image
    episode_dir.mkdir(parents=True, exist_ok=True)
    save_b64_png(ref_b64, episode_dir / "reference.png")

    # Initial conversation with reference image
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Write a GLSL shader that reproduces this reference image exactly.",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{ref_b64}",
                },
            ],
        }
    ]

    results = []

    for turn in range(1, max_turns + 1):
        print(f"  turn {turn}/{max_turns} ...", end=" ", flush=True)

        # Call GPT
        t0 = time.time()
        resp = CLIENT.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=conversation,
            max_output_tokens=8192,
            temperature=0.2,
        )
        api_s = time.time() - t0

        raw = extract_text(resp)
        code = strip_fences(raw)

        # Step the environment
        data = await ws_send(ws, "step", {"code": code})
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]

        compiled = obs["compiled"]
        rendered = obs["rendered"]
        ssim = obs["ssim"]
        errors = obs["errors"]

        # Save agent render if available
        if obs.get("agent_png"):
            save_b64_png(obs["agent_png"], episode_dir / f"turn_{turn}.png")

        turn_data = {
            "turn": turn,
            "ssim": round(ssim, 6),
            "reward": reward,
            "compiled": compiled,
            "rendered": rendered,
            "errors": errors,
            "api_seconds": round(api_s, 1),
            "code_len": len(code),
        }
        results.append(turn_data)

        if not compiled:
            status = "COMPILE_FAIL"
        elif not rendered:
            status = "RENDER_FAIL"
        else:
            status = f"ssim={ssim:.4f}"
        print(f"{status}  reward={reward}  ({api_s:.1f}s)")

        if done:
            if ssim >= 0.99:
                print(f"  => SOLVED on turn {turn}")
            else:
                print(f"  => budget exhausted")
            break

        # Feedback for next turn
        conversation.append({"role": "assistant", "content": code})

        feedback_parts = []
        if not compiled:
            feedback_parts.append(
                "Compilation FAILED.\nErrors:\n" + "\n".join(errors)
            )
        elif not rendered:
            feedback_parts.append(
                "Render FAILED.\nErrors:\n" + "\n".join(errors)
            )
        else:
            feedback_parts.append(f"SSIM: {ssim:.4f} (need >= 0.99).")
            feedback_parts.append(
                "Below is your current render vs the reference. "
                "Fix the differences. Output ONLY raw GLSL code."
            )

        feedback_content = [
            {"type": "input_text", "text": "\n".join(feedback_parts)}
        ]

        if obs.get("agent_png"):
            feedback_content.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{obs['agent_png']}"}
            )

        conversation.append({"role": "user", "content": feedback_content})

    return {
        "task": task,
        "seed": seed,
        "turns": results,
        "best_ssim": max(r["ssim"] for r in results),
        "solved": any(r["ssim"] >= 0.99 for r in results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(url: str, seeds: list[int], max_turns: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    async with websockets.connect(url) as ws:
        for i, seed in enumerate(seeds):
            label = f"episode_{i+1}"
            print(f"\n{'='*60}")
            print(f"  [{label.upper()}]  seed={seed}")
            print(f"{'='*60}")

            result = await run_episode(
                ws, seed, OUTPUT_DIR / label, max_turns,
            )
            all_results[label] = result

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for label, data in all_results.items():
        best = data["best_ssim"]
        solved = "YES" if data["solved"] else "no"
        turns_used = len(data["turns"])
        print(f"  {label}: task={data['task']}  best_ssim={best:.4f}  "
              f"solved={solved}  turns={turns_used}")

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT 5.4 on shader env")
    parser.add_argument("--url", default="ws://localhost:8000/ws",
                        help="WebSocket URL of the shader environment")
    parser.add_argument("--turns", type=int, default=10, help="Max turns per episode")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Seeds for reproducible task selection (one episode per seed)")
    args = parser.parse_args()

    asyncio.run(run(args.url, args.seeds, args.turns))


if __name__ == "__main__":
    main()
