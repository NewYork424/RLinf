# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run an OpenPI action dry-run on one frame from a real-world LeRobot dataset."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
from toolkits.eval_scripts_openpi import create_trained_policy


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _image_from_value(value: Any, root: Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value)
    if isinstance(value, dict):
        for key in ("bytes", "path"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, bytes):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return np.asarray(Image.open(path).convert("RGB"))
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        return _image_from_value(arr.item(), root)
    return arr


def _load_sample(root: Path, episode_index: int, frame_index: int) -> dict[str, Any]:
    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    if episode_index >= len(parquet_paths):
        raise IndexError(
            f"episode_index={episode_index} out of range for {len(parquet_paths)} files"
        )
    df = pd.read_parquet(parquet_paths[episode_index])
    if frame_index >= len(df):
        raise IndexError(
            f"frame_index={frame_index} out of range for episode length {len(df)}"
        )
    row = df.iloc[frame_index]
    return {
        "image": _image_from_value(row["image"], root),
        "extra_view_image": _image_from_value(row["extra_view_image"], root),
        "state": np.asarray(row["state"], dtype=np.float32),
        "task_index": int(row["task_index"]),
    }


def _task_by_index(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    return {
        int(row["task_index"]): str(row["task"])
        for row in _read_jsonl(tasks_path)
        if row.get("task_index") is not None
    }


def _summarize_actions(prompt: str, actions: np.ndarray) -> None:
    print(f"prompt: {prompt}")
    print(f"actions_shape: {list(actions.shape)}")
    print(f"finite: {bool(np.isfinite(actions).all())}")
    print(f"min: {np.min(actions, axis=0).round(5).tolist()}")
    print(f"max: {np.max(actions, axis=0).round(5).tolist()}")
    print(f"mean: {np.mean(actions, axis=0).round(5).tolist()}")
    print(f"first_action: {actions[0].round(5).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", default="pi0_realworld")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to test. Can be repeated. Defaults to the sample task prompt.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} does not contain model.safetensors. "
            "This dry-run expects an exported OpenPI checkpoint directory."
        )

    root = resolve_lerobot_dataset_root(args.repo_id)
    sample = _load_sample(root, args.episode_index, args.frame_index)
    tasks = _task_by_index(root)
    prompts = args.prompt or [tasks.get(sample["task_index"], "grasp the target object")]

    print(f"dataset_root: {root}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"sample_episode: {args.episode_index}")
    print(f"sample_frame: {args.frame_index}")
    print(f"image_shape: {list(sample['image'].shape)}")
    print(f"extra_view_image_shape: {list(sample['extra_view_image'].shape)}")
    print(f"state_shape: {list(sample['state'].shape)}")

    config = _CONFIGS_DICT[args.config_name]
    policy = create_trained_policy(
        config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )

    all_actions: list[np.ndarray] = []
    for prompt in prompts:
        policy.reset()
        observation = {
            "observation/image": sample["image"],
            "observation/extra_view_image": sample["extra_view_image"],
            "observation/state": sample["state"],
            "prompt": prompt,
        }
        actions = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
        _summarize_actions(prompt, actions)
        all_actions.append(actions)

    if len(all_actions) > 1:
        print("pairwise_action_l2:")
        for i in range(len(all_actions)):
            for j in range(i + 1, len(all_actions)):
                dist = float(np.linalg.norm(all_actions[i] - all_actions[j]))
                print(f"  {i}-{j}: {dist:.6f}")


if __name__ == "__main__":
    main()
