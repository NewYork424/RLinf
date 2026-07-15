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

"""Check a real-world Franka grasp LeRobot dataset before SFT.

This script is intentionally conservative: it fails on missing camera/state/action
features, non-finite values, all-zero action episodes, and missing task prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root


REQUIRED_FEATURES = ("image", "extra_view_image", "state", "actions", "task_index")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _feature_shape(info: dict[str, Any], key: str) -> list[int] | None:
    feature = info.get("features", {}).get(key, {})
    shape = feature.get("shape")
    return list(shape) if shape is not None else None


def _iter_episode_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet"))


def _as_array(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        return np.empty((0,))
    try:
        return np.stack(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def check_dataset(repo_id: str, min_nonzero_action_ratio: float) -> int:
    root = resolve_lerobot_dataset_root(repo_id)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    parquet_paths = _iter_episode_parquets(root)

    print(f"dataset: {repo_id}")
    print(f"root: {root}")
    print(f"parquet_files: {len(parquet_paths)}")
    print(f"features: {sorted(info.get('features', {}).keys())}")
    print(f"tasks: {[row.get('task') for row in tasks]}")

    errors: list[str] = []
    for key in REQUIRED_FEATURES:
        if key not in info.get("features", {}):
            errors.append(f"missing required feature: {key}")

    action_shape = _feature_shape(info, "actions")
    state_shape = _feature_shape(info, "state")
    image_shape = _feature_shape(info, "image")
    extra_shape = _feature_shape(info, "extra_view_image")
    print(f"state_shape: {state_shape}")
    print(f"action_shape: {action_shape}")
    print(f"image_shape: {image_shape}")
    print(f"extra_view_image_shape: {extra_shape}")

    if action_shape != [7]:
        errors.append(f"actions shape should be [7], got {action_shape}")
    if state_shape is None or len(state_shape) != 1:
        errors.append(f"state shape should be 1D, got {state_shape}")
    if image_shape is None or image_shape[-1:] != [3]:
        errors.append(f"image shape should end with 3 channels, got {image_shape}")
    if extra_shape is None or extra_shape[-1:] != [3]:
        errors.append(
            f"extra_view_image shape should end with 3 channels, got {extra_shape}"
        )
    if not tasks:
        errors.append("missing meta/tasks.jsonl or no task prompts recorded")

    total_rows = 0
    total_nonzero = 0
    task_indices: set[int] = set()
    episode_summaries = []

    for path in parquet_paths:
        df = pd.read_parquet(path)
        missing_cols = [key for key in REQUIRED_FEATURES if key not in df.columns]
        if missing_cols:
            errors.append(f"{path}: missing columns {missing_cols}")
            continue

        actions = _as_array(df["actions"])
        states = _as_array(df["state"])
        nonzero_mask = np.any(np.abs(actions) > 1e-8, axis=-1)
        finite_actions = np.isfinite(actions).all()
        finite_states = np.isfinite(states).all()
        nonzero = int(nonzero_mask.sum())
        rows = len(df)
        ratio = float(nonzero / max(rows, 1))
        total_rows += rows
        total_nonzero += nonzero
        task_indices.update(int(x) for x in df["task_index"].dropna().unique())

        rel_path = path.relative_to(root)
        episode_summaries.append((str(rel_path), rows, nonzero, ratio))
        if not finite_actions:
            errors.append(f"{rel_path}: actions contain NaN or Inf")
        if not finite_states:
            errors.append(f"{rel_path}: state contains NaN or Inf")
        if ratio < min_nonzero_action_ratio:
            errors.append(
                f"{rel_path}: nonzero action ratio {ratio:.3f} "
                f"< {min_nonzero_action_ratio:.3f}"
            )

    print("episodes:")
    for rel_path, rows, nonzero, ratio in episode_summaries:
        print(f"  {rel_path}: rows={rows}, nonzero_actions={nonzero}, ratio={ratio:.3f}")

    print(f"total_rows: {total_rows}")
    print(f"total_nonzero_actions: {total_nonzero}")
    print(f"task_indices_in_data: {sorted(task_indices)}")

    task_index_set = {
        int(row["task_index"]) for row in tasks if row.get("task_index") is not None
    }
    missing_task_defs = task_indices - task_index_set
    if missing_task_defs:
        errors.append(f"task indices missing from tasks.jsonl: {sorted(missing_task_defs)}")

    if errors:
        print("FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--min-nonzero-action-ratio", type=float, default=0.05)
    args = parser.parse_args()
    raise SystemExit(check_dataset(args.repo_id, args.min_nonzero_action_ratio))


if __name__ == "__main__":
    main()
