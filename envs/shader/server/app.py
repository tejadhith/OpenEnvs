"""
FastAPI application for the shader environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from ..models import ShaderAction, ShaderObservation
    from .environment import ShaderEnvironment
except ImportError:
    from models import ShaderAction, ShaderObservation
    from server.environment import ShaderEnvironment

app = create_app(
    ShaderEnvironment,
    ShaderAction,
    ShaderObservation,
    env_name="shader",
    max_concurrent_envs=4,
)


def main() -> None:
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
