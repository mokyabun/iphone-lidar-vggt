#!/usr/bin/env python3
"""Create Vast.ai templates for the iPhone LiDAR + ReconViaGen server."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import textwrap
from dataclasses import dataclass


DEFAULT_REPO = "https://github.com/mokyabun/iphone-lidar-vggt.git"
DEFAULT_BRANCH = "main"
DEFAULT_DISK_GB = 120
DEFAULT_SEARCH_PARAMS = "compute_cap >= 750 cuda_max_good >= 12.1 gpu_ram >= 24000"


@dataclass(frozen=True)
class Template:
    key: str
    name: str
    image: str
    image_tag: str
    description: str
    use_system_torch: bool


TEMPLATES = (
    Template(
        key="torch",
        name="iPhone LiDAR ReconViaGen - Vast PyTorch",
        image="vastai/pytorch",
        image_tag="2.4.1-cuda-12.4.1-22.04",
        description="Runs this project on Vast's PyTorch image and reuses the image-provided torch.",
        use_system_torch=True,
    ),
    Template(
        key="cuda",
        name="iPhone LiDAR ReconViaGen - Vast CUDA",
        image="vastai/base-image",
        image_tag="@vastai-automatic-tag",
        description="Runs this project on Vast's CUDA base image; uv installs pinned Torch packages.",
        use_system_torch=False,
    ),
)


def shell_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def docker_options(use_system_torch: bool) -> str:
    portal_config = (
        "localhost:1111:11111:/:Instance Portal|"
        "localhost:8000:8000:/docs:LiDAR API|"
        "localhost:8011:8011:/health:ReconViaGen Worker|"
        "localhost:8080:18080:/:Jupyter|"
        "localhost:8080:8080:/terminals/1:Jupyter Terminal|"
        "localhost:8384:18384:/:Syncthing"
    )
    env = {
        "OPEN_BUTTON_PORT": "1111",
        "OPEN_BUTTON_TOKEN": "1",
        "JUPYTER_DIR": "/",
        "DATA_DIRECTORY": "/workspace/",
        "PORTAL_CONFIG": portal_config,
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "8000",
        "APP_CACHE_ROOT": "/workspace/cache",
        "RECONVIAGEN_WORKER_HOST": "0.0.0.0",
        "RECONVIAGEN_WORKER_PORT": "8011",
        "RECONVIAGEN_USE_SYSTEM_TORCH": "1" if use_system_torch else "0",
        "RECONVIAGEN_PYTHON_VERSION": "python3" if use_system_torch else "3.10",
        "HF_TOKEN": "",
    }
    parts = [
        "-p 1111:1111",
        "-p 8000:8000",
        "-p 8011:8011",
        "-p 8080:8080",
        "-p 8384:8384",
        "-p 72299:72299",
        "-p 10100:10100",
        "-p 10200:10200",
    ]
    for key, value in env.items():
        parts.append(f"-e {key}={shell_value(value)}")
    return " ".join(parts)


def onstart_script(repo: str, branch: str) -> str:
    return textwrap.dedent(
        f"""\
        set -Eeuo pipefail

        export DEBIAN_FRONTEND=noninteractive
        if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
          apt-get update
          apt-get install -y git curl ca-certificates build-essential
        fi

        export PROJECT_REPO="${{PROJECT_REPO:-{repo}}}"
        export PROJECT_BRANCH="${{PROJECT_BRANCH:-{branch}}}"
        export APP_CACHE_ROOT="${{APP_CACHE_ROOT:-/workspace/cache}}"
        export APP_HOST="${{APP_HOST:-0.0.0.0}}"
        export APP_PORT="${{APP_PORT:-8000}}"
        export RECONVIAGEN_WORKER_HOST="${{RECONVIAGEN_WORKER_HOST:-0.0.0.0}}"
        export RECONVIAGEN_WORKER_PORT="${{RECONVIAGEN_WORKER_PORT:-8011}}"

        mkdir -p /workspace "${{APP_CACHE_ROOT}}"
        cd /workspace

        if [ ! -d iphone-lidar-vggt/.git ]; then
          rm -rf iphone-lidar-vggt
          git clone --branch "${{PROJECT_BRANCH}}" "${{PROJECT_REPO}}" iphone-lidar-vggt
        fi

        cd iphone-lidar-vggt
        git fetch origin "${{PROJECT_BRANCH}}"
        git checkout "${{PROJECT_BRANCH}}"
        git reset --hard "origin/${{PROJECT_BRANCH}}"

        if [ -n "${{HF_TOKEN:-}}" ] && [ -z "${{HUGGINGFACE_HUB_TOKEN:-}}" ]; then
          export HUGGINGFACE_HUB_TOKEN="${{HF_TOKEN}}"
        fi

        cd server
        chmod +x ./run.sh ./scripts/install/uv.sh
        exec ./run.sh
        """
    )


def build_command(args: argparse.Namespace, template: Template) -> list[str]:
    image = getattr(args, f"{template.key}_image")
    image_tag = getattr(args, f"{template.key}_tag")
    name = f"{args.name_prefix} {template.name}" if args.name_prefix else template.name
    desc = f"{template.description} Exposes API :8000, worker :8011, and SSH direct."

    cmd = list(args.vastai_cmd)
    cmd.extend(
        [
            "create",
            "template",
            "--name",
            name,
            "--image",
            image,
            "--image_tag",
            image_tag,
            "--href",
            args.repo,
            "--repo",
            args.repo,
            "--env",
            docker_options(template.use_system_torch),
            "--ssh",
            "--direct",
            "--onstart-cmd",
            onstart_script(args.repo, args.branch),
            "--disk_space",
            str(args.disk_gb),
            "--desc",
            desc,
            "--readme",
            readme_text(args.repo, args.branch),
        ]
    )
    if args.search_params:
        cmd.extend(["--search_params", args.search_params])
    if args.public:
        cmd.append("--public")
    return cmd


def readme_text(repo: str, branch: str) -> str:
    return textwrap.dedent(
        f"""\
        Starts the iPhone LiDAR + ReconViaGen backend from {repo} ({branch}).

        Exposed services:
        - API: port 8000, FastAPI docs at /docs
        - ReconViaGen worker: port 8011, health at /health
        - SSH: enabled with direct connections

        Set HF_TOKEN on the instance if gated Hugging Face weights are needed.
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create two Vast.ai templates for this project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call Vast.ai. Without this, commands are printed only.",
    )
    parser.add_argument(
        "--vastai-cmd",
        default="uv tool run vastai",
        help="Command used to invoke Vast.ai CLI.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Git repository cloned on instance start.")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch checked out on instance start.")
    parser.add_argument("--name-prefix", default="", help="Optional prefix added to each template name.")
    parser.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB, help="Recommended template disk size.")
    parser.add_argument(
        "--search-params",
        default=DEFAULT_SEARCH_PARAMS,
        help="Default Vast offer filters saved on the template. Use '' to omit.",
    )
    parser.add_argument("--torch-image", default=TEMPLATES[0].image, help="Docker image for the PyTorch template.")
    parser.add_argument("--torch-tag", default=TEMPLATES[0].image_tag, help="Docker tag for the PyTorch template.")
    parser.add_argument("--cuda-image", default=TEMPLATES[1].image, help="Docker image for the CUDA template.")
    parser.add_argument("--cuda-tag", default=TEMPLATES[1].image_tag, help="Docker tag for the CUDA template.")
    parser.add_argument("--public", action="store_true", help="Make the templates public.")
    args = parser.parse_args()
    args.vastai_cmd = shlex.split(args.vastai_cmd)
    return args


def main() -> int:
    args = parse_args()
    for template in TEMPLATES:
        cmd = build_command(args, template)
        print(shlex.join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
