
# 这是 inspect_franka_video.py 的扩展版本：
#   - 完全保留原有功能（meta / parquet / 长度 / 文件大小 / 索引一致性 检查 + 视频导出）
#   - 新增「缺失文件诊断」功能：
#       1) 检查每个 id 下 data/ 和 meta/ 目录本身是否存在（缺失标 CRITICAL）
#          即使 id 目录不完整也会被扫描（不再因缺 data/meta 而跳过）
#       2) 检查 meta/ 目录下应有的关键文件是否缺失
#          (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
#       3) 检查 data/ 目录下 parquet 文件是否缺失：
#            - episode_NNNNNN.parquet 编号是否有空缺 (gap)
#            - episodes.jsonl / episodes_stats.jsonl 中声明、但磁盘上不存在的 episode_index
#            - 0 字节 parquet
#       4) 在控制台打印 [MISSING-FILES] 报告，并写入 summary.json 中的 missing_files_check 字段
#
# 用法与原脚本一致，例如：
#   python inspect_franka_video_with_missing_check.py --root <ROOT> --out_dir <OUT> --num_vis 0 --save_last_frame
#
# 单个 id:
# python3 inspect_franka_video_with_missing_check.py --root /home/raojiaji/data_raojiaji/rollout_data/69_rollout_hold_bag_data/rank_0/id_0 --out_dir franka_inspect_outputs_rollout --num_vis 1 --seed 42
# 扫描多个 id:
# python3 /home/raojiaji/Jiahao/RLinf/inspect11.py --root /home/raojiaji/data_raojiaji/0624_sby_pull_drawer --out_dir pull_inspect_outputs --num_vis 0 --seed 42

import json
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# 以下区块（直到 "NEW: missing-files diagnostic" 之前）来自原始
# inspect_franka_video.py，逻辑保持不变。
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def is_scalar_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

def is_list_like(x):
    return isinstance(x, (list, tuple, np.ndarray))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path, max_lines=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_lines is not None and i + 1 >= max_lines:
                break
    return rows

def count_jsonl_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def stream_jsonl_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_episodes_idx_to_length(path):
    out = {}
    for obj in stream_jsonl_rows(path):
        ei = obj.get("episode_index")
        ln = obj.get("length")
        if isinstance(ei, int):
            out[ei] = ln if isinstance(ln, int) else None
    return out

def load_episode_indices(path):
    out = set()
    for obj in stream_jsonl_rows(path):
        ei = obj.get("episode_index")
        if isinstance(ei, int):
            out.add(ei)
    return sorted(out)

def maybe_to_1d_numeric_array(series):
    vals = series.tolist()
    out = []
    for v in vals:
        if v is None:
            out.append(np.nan)
        elif is_scalar_number(v):
            out.append(float(v))
        else:
            return None
    return np.asarray(out, dtype=np.float64)

def maybe_to_2d_numeric_array(series):
    vals = series.tolist()
    rows = []
    expected_len = None

    for v in vals:
        if v is None or not is_list_like(v):
            return None

        arr = np.asarray(v)

        if arr.ndim != 1:
            return None

        if not np.issubdtype(arr.dtype, np.number):
            try:
                arr = arr.astype(np.float64)
            except Exception:
                return None

        if expected_len is None:
            expected_len = arr.shape[0]
        if arr.shape[0] != expected_len:
            return None

        rows.append(arr.astype(np.float64))

    if not rows:
        return None

    return np.stack(rows, axis=0)

def flatten_dict(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            items.update(flatten_dict(v, name))
    else:
        items[prefix] = obj
    return items

def summarize_meta(meta_dir):
    report = {
        "meta_exists": meta_dir.exists(),
        "info": {},
        "episodes_sample": [],
        "episodes_stats_sample": [],
        "tasks_sample": [],
        "feature_keys": [],
        "image_feature_candidates": [],
        "num_episodes_jsonl": None,
        "num_episodes_stats_jsonl": None,
        "episodes_index_to_length": {},
        "episodes_stats_episode_indices": [],
        "errors": [],
    }

    if not meta_dir.exists():
        report["errors"].append(f"meta dir not found: {meta_dir}")
        return report

    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    if info_path.exists():
        try:
            info = load_json(info_path)
            report["info"] = info

            feature_keys = []
            image_candidates = []

            flat = flatten_dict(info)
            for k, v in flat.items():
                if "feature" in k.lower() or "column" in k.lower() or "schema" in k.lower():
                    feature_keys.append(k)

            candidate_containers = []
            for key in ["features", "observation", "schema", "columns"]:
                if key in info and isinstance(info[key], dict):
                    candidate_containers.append(info[key])

            for container in candidate_containers:
                for k, v in container.items():
                    kl = k.lower()
                    vv = str(v).lower()
                    if any(w in kl for w in ["image", "rgb", "camera", "cam", "view"]):
                        image_candidates.append(k)
                    elif any(w in vv for w in ["image", "rgb", "camera", "cam", "view"]):
                        image_candidates.append(k)

            report["feature_keys"] = sorted(set(feature_keys))
            report["image_feature_candidates"] = sorted(set(image_candidates))
        except Exception as e:
            report["errors"].append(f"failed to read info.json: {e}")

    if episodes_path.exists():
        try:
            report["episodes_sample"] = load_jsonl(episodes_path, max_lines=5)
            report["num_episodes_jsonl"] = count_jsonl_lines(episodes_path)
            report["episodes_index_to_length"] = load_episodes_idx_to_length(episodes_path)
        except Exception as e:
            report["errors"].append(f"failed to read episodes.jsonl: {e}")

    if episodes_stats_path.exists():
        try:
            report["episodes_stats_sample"] = load_jsonl(episodes_stats_path, max_lines=5)
            report["num_episodes_stats_jsonl"] = count_jsonl_lines(episodes_stats_path)
            report["episodes_stats_episode_indices"] = load_episode_indices(episodes_stats_path)
        except Exception as e:
            report["errors"].append(f"failed to read episodes_stats.jsonl: {e}")

    if tasks_path.exists():
        try:
            report["tasks_sample"] = load_jsonl(tasks_path, max_lines=5)
        except Exception as e:
            report["errors"].append(f"failed to read tasks.jsonl: {e}")

    return report

def decode_image_cell(cell, dataset_root=None):
    def _unwrap(x, max_depth=8):
        cur = x
        for _ in range(max_depth):
            if cur is None:
                return None

            if isinstance(cur, np.ndarray) and cur.dtype == object and cur.shape == ():
                cur = cur.item()
                continue

            if isinstance(cur, dict):
                for k in ["array", "image", "bytes", "data", "value", "frame", "path"]:
                    if k in cur:
                        cur = cur[k]
                        break
                else:
                    return cur
                continue

            return cur
        return cur

    cell = _unwrap(cell)
    if cell is None:
        return None

    if isinstance(cell, bytes):
        buf = np.frombuffer(cell, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
        return None

    if isinstance(cell, str):
        p = Path(cell)
        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            if dataset_root is not None:
                candidates.append(Path(dataset_root) / cell)
                candidates.append(Path(dataset_root) / "data" / cell)
            candidates.append(p)

        for cand in candidates:
            if cand.exists() and cand.is_file():
                try:
                    img = Image.open(cand).convert("RGB")
                    return np.asarray(img)
                except Exception:
                    try:
                        img = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            return img
                    except Exception:
                        pass
        return None

    if isinstance(cell, (list, tuple)):
        cell = np.asarray(cell)

    if isinstance(cell, np.ndarray):
        arr = cell

        if arr.dtype == object and arr.shape == ():
            return decode_image_cell(arr.item(), dataset_root=dataset_root)

        if arr.ndim == 1 and arr.dtype == np.uint8:
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
            return arr

    try:
        arr = np.asarray(cell)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
            return arr
    except Exception:
        pass

    return None

def can_decode_series_as_image(series, dataset_root=None, num_samples=8):
    vals = series.tolist()
    if len(vals) == 0:
        return False

    sample_indices = np.linspace(0, len(vals) - 1, num=min(num_samples, len(vals)), dtype=int)
    ok = 0
    for idx in sample_indices:
        if decode_image_cell(vals[int(idx)], dataset_root=dataset_root) is not None:
            ok += 1
    return ok > 0

def choose_image_columns(df, meta_report, dataset_root=None):
    meta_candidates = []
    if meta_report:
        meta_candidates = [c for c in meta_report.get("image_feature_candidates", []) if c in df.columns]
    if meta_candidates:
        return sorted(set(meta_candidates))

    detected = []
    for col in df.columns:
        col_l = col.lower()
        if any(w in col_l for w in ["image", "rgb", "camera", "cam", "view"]):
            if can_decode_series_as_image(df[col], dataset_root=dataset_root):
                detected.append(col)
    if detected:
        return detected

    for col in df.columns:
        if can_decode_series_as_image(df[col], dataset_root=dataset_root):
            detected.append(col)
    return detected

def to_uint8_image(img):
    arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.size == 0:
        return None

    vmax = arr.max()
    vmin = arr.min()

    if vmax <= 1.0 and vmin >= 0.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def to_bgr_image(img):
    if img is None:
        return None

    img = to_uint8_image(img)
    if img is None:
        return None

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if img.ndim == 3 and img.shape[-1] == 3:
        return img.copy()

    return None

def resize_with_pad(img, target_h, target_w):
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def draw_text_bar(width, text, height=32):
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar

def annotate_panel(img, label):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def summarize_episode(parquet_path, meta_report, dataset_root):
    result = {
        "file": str(parquet_path),
        "size_bytes": parquet_path.stat().st_size,
        "load_ok": False,
        "num_rows": 0,
        "issues": [],
        "columns": [],
        "image_like_columns": [],
        "state_shape": None,
        "actions_shape": None,
        "timestamp_range": None,
    }

    if parquet_path.stat().st_size == 0:
        result["issues"].append("empty parquet file")
        return result

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        result["issues"].append(f"load failed: {e}")
        return result

    result["load_ok"] = True
    result["num_rows"] = int(len(df))
    result["columns"] = list(df.columns)
    result["image_like_columns"] = choose_image_columns(df, meta_report, dataset_root=dataset_root)

    if "state" in df.columns:
        arr = maybe_to_2d_numeric_array(df["state"])
        if arr is not None:
            result["state_shape"] = list(arr.shape)
        else:
            result["issues"].append("state column not numeric/list-like")

    if "actions" in df.columns:
        arr = maybe_to_2d_numeric_array(df["actions"])
        if arr is not None:
            result["actions_shape"] = list(arr.shape)
        else:
            result["issues"].append("actions column not numeric/list-like")

    if "timestamp" in df.columns:
        ts = maybe_to_1d_numeric_array(df["timestamp"])
        if ts is not None and len(ts) > 0:
            result["timestamp_range"] = [float(np.min(ts)), float(np.max(ts))]
            dt = np.diff(ts)
            if len(dt) > 0 and np.any(dt < 0):
                result["issues"].append("timestamp not monotonic")
            if np.any(dt == 0):
                result["issues"].append("timestamp duplicated")

    return result

def build_summary(reports):
    summary = {
        "num_episodes": len(reports),
        "load_failed": 0,
        "episodes_with_issues": 0,
        "issue_counter": defaultdict(int),
    }

    for r in reports:
        if not r["load_ok"]:
            summary["load_failed"] += 1
        if r["issues"]:
            summary["episodes_with_issues"] += 1
            for issue in r["issues"]:
                summary["issue_counter"][issue] += 1

    summary["issue_counter"] = dict(summary["issue_counter"])
    return summary

def build_video_frame(df, frame_idx, image_cols, dataset_root, target_h=480, target_w=640):
    panels = []
    for col in image_cols:
        cell = df.iloc[frame_idx][col]
        img = decode_image_cell(cell, dataset_root=dataset_root)
        bgr = to_bgr_image(img)
        if bgr is None:
            bgr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        bgr = resize_with_pad(bgr, target_h, target_w)
        bgr = annotate_panel(bgr, col)
        panels.append(bgr)

    if not panels:
        return None

    row = np.concatenate(panels, axis=1)

    timestamp = None
    if "timestamp" in df.columns:
        try:
            timestamp = float(df.iloc[frame_idx]["timestamp"])
        except Exception:
            timestamp = None

    segment_id = None
    if "segment_id" in df.columns:
        segment_id = df.iloc[frame_idx]["segment_id"]

    meta_text = f"frame {frame_idx:06d}"
    if timestamp is not None:
        meta_text += f"  t={timestamp:.3f}s"
    if segment_id is not None:
        meta_text += f"  segment={segment_id}"

    top_bar = draw_text_bar(row.shape[1], meta_text)
    return np.concatenate([top_bar, row], axis=0)

def export_episode_video(parquet_path, out_path, meta_report, dataset_root,
                          save_video=True, save_last_frame=False):
    df = pd.read_parquet(parquet_path)
    image_cols = choose_image_columns(df, meta_report, dataset_root=dataset_root)
    print(f"[DEBUG] image columns for {parquet_path.name}: {image_cols}")

    if not image_cols:
        return {
            "file": str(parquet_path),
            "video": None,
            "image_columns": [],
            "last_frame": None,
            "error": "no decodable image columns detected",
        }

    for col in image_cols:
        first_ok = None
        for i in range(min(len(df), 10)):
            if decode_image_cell(df.iloc[i][col], dataset_root=dataset_root) is not None:
                first_ok = i
                break
        print(f"[DEBUG] {col}: first decodable frame = {first_ok}")

    fps = 10.0
    if "timestamp" in df.columns:
        ts = maybe_to_1d_numeric_array(df["timestamp"])
        if ts is not None and len(ts) >= 2:
            dt = np.diff(ts)
            mean_dt = np.mean(dt)
            if mean_dt > 1e-8:
                fps = float(1.0 / mean_dt)

    first_frame = build_video_frame(df, 0, image_cols, dataset_root=dataset_root)
    if first_frame is None:
        return {
            "file": str(parquet_path),
            "video": None,
            "image_columns": image_cols,
            "last_frame": None,
            "error": "failed to build first video frame",
        }

    h, w = first_frame.shape[:2]

    last_frame = None
    video_path = None

    if save_video:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if not writer.isOpened():
            return {
                "file": str(parquet_path),
                "video": None,
                "image_columns": image_cols,
                "last_frame": None,
                "error": "cv2.VideoWriter failed to open",
            }

        for i in range(len(df)):
            frame = build_video_frame(df, i, image_cols, dataset_root=dataset_root)
            if frame is not None:
                writer.write(frame)
                last_frame = frame

        writer.release()
        video_path = str(out_path)
        print(f"[INFO] saved video to {out_path}")

    if save_last_frame and last_frame is None and len(df) > 0:
        last_frame = build_video_frame(df, len(df) - 1, image_cols, dataset_root=dataset_root)

    last_frame_path = None
    if save_last_frame and last_frame is not None:
        last_frame_path = out_path.with_suffix(".last_frame.jpg")
        frame_to_save = last_frame

        if len(frame_to_save.shape) == 3 and frame_to_save.shape[2] == 4:
            frame_to_save = cv2.cvtColor(frame_to_save, cv2.COLOR_BGRA2BGR)
        elif len(frame_to_save.shape) == 2:
            frame_to_save = cv2.cvtColor(frame_to_save, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(last_frame_path), frame_to_save)
        last_frame_path = str(last_frame_path)
        print(f"[INFO] saved last frame to {last_frame_path}")

    return {
        "file": str(parquet_path),
        "video": video_path,
        "image_columns": image_cols,
        "fps": fps if save_video else None,
        "last_frame": last_frame_path,
        "error": None,
    }

_EP_INDEX_RE = re.compile(r"episode_(\d+)\.parquet$")
# |z| > threshold 判为 parquet 大小异常
PARQUET_SIZE_Z_THRESHOLD = 3.0


def check_parquet_sizes(episode_files, z_threshold=PARQUET_SIZE_Z_THRESHOLD):
    """
    检测 parquet 文件大小异常（z-score 方法）。

    z = (size - mean) / std
    - z < -threshold → 文件过小 (too_small)
    - z >  threshold → 文件过大 (too_large)
    """
    sizes = [(ep, ep.stat().st_size) for ep in episode_files if ep.exists()]
    if not sizes:
        return {
            "num_files": 0,
            "anomalous_files": [],
            "too_small_files": [],
            "too_large_files": [],
            "error": "no files",
        }

    arr = np.array([s for _, s in sizes], dtype=float)
    mean, std = float(np.mean(arr)), float(np.std(arr))

    base = {
        "num_files": len(sizes),
        "mean_bytes": round(mean, 1),
        "std_bytes": round(std, 1),
        "median_bytes": round(float(np.median(arr)), 1),
        "z_score_threshold": z_threshold,
    }

    if std <= 1e-8:
        return {
            **base,
            "num_anomalous": 0,
            "num_too_small": 0,
            "num_too_large": 0,
            "anomalous_files": [],
            "too_small_files": [],
            "too_large_files": [],
            "note": "all files have identical size, z-score check skipped",
        }

    too_small = []
    too_large = []
    anomalous = []

    for ep, sz in sizes:
        z = (sz - mean) / std
        if z < -z_threshold:
            entry = {
                "file": str(ep.name),
                "size_bytes": int(sz),
                "z_score": round(z, 3),
                "issue": "too_small",
            }
            too_small.append(entry)
            anomalous.append(entry)
        elif z > z_threshold:
            entry = {
                "file": str(ep.name),
                "size_bytes": int(sz),
                "z_score": round(z, 3),
                "issue": "too_large",
            }
            too_large.append(entry)
            anomalous.append(entry)

    return {
        **base,
        "num_anomalous": len(anomalous),
        "num_too_small": len(too_small),
        "num_too_large": len(too_large),
        "anomalous_files": anomalous,
        "too_small_files": too_small,
        "too_large_files": too_large,
    }

def extract_episode_index(path):
    m = _EP_INDEX_RE.search(str(path))
    return int(m.group(1)) if m else None

def find_episode_files(data_dir):
    return sorted(data_dir.rglob("*.parquet"))

def discover_id_roots(root):
    """Find dataset id directories to inspect.

    Returns every ``id_*`` subdirectory under ``root``, including those
    missing ``data/`` or ``meta/`` so that directory-level gaps are reported.
    If ``root`` itself is a single-id layout (has ``data/`` + ``meta/``), or
    is passed directly as an incomplete id dir, ``root`` is returned as-is.
    """
    root = Path(root)

    if (root / "data").exists() and (root / "meta").exists():
        return [root]

    id_roots = sorted(p for p in root.rglob("id_*") if p.is_dir())
    if id_roots:
        return id_roots

    if root.is_dir():
        return [root]

    return []

# ============================================================
# NEW: missing-files diagnostic
# 仅用于检测「应有但磁盘上不存在 / 为空」的 meta 与 data 文件，
# 不修改原有流程，只在 process_one_id 的返回值里多加一个字段，
# 在 main() 里多打印一段 [MISSING-FILES] 报告。
# ============================================================

# 期望的 meta 文件清单。如果你的数据集还有别的必备 meta 文件，
# 在这里加进来就会被一并检查。
EXPECTED_META_FILES = [
    "info.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "tasks.jsonl",
]

# 这些 meta 文件被认为是「核心 / 必须」的，缺失会被标记成 critical。
# 其余的（例如 tasks.jsonl）当作 optional，仅作为提示。
CRITICAL_META_FILES = {"info.json", "episodes.jsonl"}


def diagnose_missing_files(id_root, episode_files, meta_report):
    """
    诊断一个 id 目录下 meta / data 文件的缺失情况。

    Args:
        id_root: 该 id 的根目录 (Path)，下面应有 data/ 与 meta/。
        episode_files: 该 id 下 sorted 后的 parquet 文件列表 (list[Path])。
        meta_report: summarize_meta(meta_dir) 的返回值，用来判断 meta 中
                     声明了哪些 episode_index。

    Returns:
        dict: 详细诊断结果，结构见下方注释。
    """
    id_root = Path(id_root)
    data_dir = id_root / "data"
    meta_dir = id_root / "meta"

    data_exists = data_dir.is_dir()
    meta_exists = meta_dir.is_dir()
    missing_directories = []
    if not data_exists:
        missing_directories.append("data")
    if not meta_exists:
        missing_directories.append("meta")

    report = {
        "id_root": str(id_root),
        "data_dir_exists": data_exists,
        "meta_dir_exists": meta_exists,
        "missing_directories": missing_directories,

        # meta 文件缺失情况
        "meta_files_status": {},          # filename -> {"exists": bool, "size_bytes": int|None, "is_empty": bool}
        "missing_meta_files": [],         # 不存在的 meta 文件
        "empty_meta_files": [],           # 存在但 0 字节的 meta 文件
        "missing_critical_meta_files": [],  # CRITICAL_META_FILES 里缺失的

        # data 文件缺失情况
        "num_parquet_files": len(episode_files),
        "empty_parquet_files": [],            # 存在但 0 字节的 parquet
        "missing_parquet_indices_by_gap": [],   # 0..max 中编号不连续而推断缺失的
        "missing_parquet_from_episodes_jsonl": [],         # episodes.jsonl 声明但磁盘缺失
        "missing_parquet_from_episodes_stats_jsonl": [],   # episodes_stats.jsonl 声明但磁盘缺失
        "missing_parquet_from_declared_total": [],         # info.json 声明 total_episodes 但磁盘缺失
        "missing_parquet_indices_union": [],  # 上述各来源的并集，一站式查看

        # 顶层 OK 标志
        "is_ok": True,
        "summary_message": "",
    }

    # ----- 1. 检查 data/ 与 meta/ 目录本身 -----
    if missing_directories:
        report["is_ok"] = False

    # ----- 2. 检查 meta 文件（仅当 meta/ 目录存在时） -----
    if not meta_exists:
        for fname in EXPECTED_META_FILES:
            report["meta_files_status"][fname] = {
                "exists": False,
                "size_bytes": None,
                "is_empty": False,
                "skipped": "meta/ directory missing",
            }
    else:
        for fname in EXPECTED_META_FILES:
            fpath = meta_dir / fname
            exists = fpath.exists() and fpath.is_file()
            size_bytes = int(fpath.stat().st_size) if exists else None
            is_empty = bool(exists and size_bytes == 0)

            report["meta_files_status"][fname] = {
                "exists": exists,
                "size_bytes": size_bytes,
                "is_empty": is_empty,
            }

            if not exists:
                report["missing_meta_files"].append(fname)
                if fname in CRITICAL_META_FILES:
                    report["missing_critical_meta_files"].append(fname)
                    report["is_ok"] = False
            elif is_empty:
                report["empty_meta_files"].append(fname)
                if fname in CRITICAL_META_FILES:
                    report["missing_critical_meta_files"].append(fname)
                    report["is_ok"] = False

    # ----- 3. 检查 data parquet 文件（data/ 不存在时 episode_files 为空） -----
    parquet_index_to_path = {}
    for ep in episode_files:
        idx = extract_episode_index(ep)
        if idx is not None:
            parquet_index_to_path[idx] = ep
        try:
            if ep.stat().st_size == 0:
                report["empty_parquet_files"].append(str(ep.name))
                report["is_ok"] = False
        except OSError:
            # 文件出现在 listing 但 stat 失败，按缺失处理
            report["empty_parquet_files"].append(str(ep.name))
            report["is_ok"] = False

    found_set = set(parquet_index_to_path.keys())

    # 3a) 用编号 gap 推断缺失：从 0 到 max(found_set) 之间应该都存在
    gap_missing = []
    if found_set:
        for i in range(min(found_set), max(found_set) + 1):
            if i not in found_set:
                gap_missing.append(i)
    report["missing_parquet_indices_by_gap"] = gap_missing
    if gap_missing:
        report["is_ok"] = False

    # 3b) episodes.jsonl 声明了哪些 index：磁盘缺失就记录
    declared_eps_idx = set((meta_report or {}).get("episodes_index_to_length", {}) or {})
    missing_from_eps = sorted(declared_eps_idx - found_set)
    report["missing_parquet_from_episodes_jsonl"] = missing_from_eps
    if missing_from_eps:
        report["is_ok"] = False

    # 3c) episodes_stats.jsonl 声明了哪些 index：磁盘缺失就记录
    declared_eps_stats_idx = set((meta_report or {}).get("episodes_stats_episode_indices", []) or [])
    missing_from_eps_stats = sorted(declared_eps_stats_idx - found_set)
    report["missing_parquet_from_episodes_stats_jsonl"] = missing_from_eps_stats
    if missing_from_eps_stats:
        report["is_ok"] = False

    # 3d) info.json 中的 total_episodes / num_episodes 声明
    info = (meta_report or {}).get("info") or {}
    declared_total = None
    for k in ("total_episodes", "num_episodes"):
        v = info.get(k)
        if isinstance(v, int):
            declared_total = v
            break

    missing_from_declared_total = []
    if declared_total is not None:
        # 假定 episode_index 是 [0, declared_total)
        missing_from_declared_total = [i for i in range(declared_total) if i not in found_set]
    report["missing_parquet_from_declared_total"] = missing_from_declared_total
    if missing_from_declared_total:
        report["is_ok"] = False

    # 3e) 把所有「应该有但缺失」的 episode_index 汇总成一个并集
    union = set()
    union.update(gap_missing)
    union.update(missing_from_eps)
    union.update(missing_from_eps_stats)
    union.update(missing_from_declared_total)
    report["missing_parquet_indices_union"] = sorted(union)

    # ----- 4. 顶层概要消息 -----
    parts = []
    if "data" in missing_directories:
        parts.append("data/ MISSING (CRITICAL)")
    if "meta" in missing_directories:
        parts.append("meta/ MISSING (CRITICAL)")
    if report["missing_meta_files"]:
        parts.append(f"missing meta files: {report['missing_meta_files']}")
    if report["empty_meta_files"]:
        parts.append(f"empty meta files: {report['empty_meta_files']}")
    if report["empty_parquet_files"]:
        parts.append(f"empty parquet: {len(report['empty_parquet_files'])}")
    if report["missing_parquet_indices_union"]:
        parts.append(f"missing parquet indices (union): {len(report['missing_parquet_indices_union'])}")

    report["summary_message"] = "OK" if not parts else "; ".join(parts)
    return report


def print_missing_files_report(id_root, mf):
    """打印一段易读的 [MISSING-FILES] 报告。"""
    name = Path(id_root).name
    tag = "OK" if mf["is_ok"] else "ISSUES"
    print(f"[MISSING-FILES] {name}: [{tag}] {mf['summary_message']}")

    for dirname in mf.get("missing_directories", []):
        print(f"[MISSING-FILES] {name}: {dirname}/ directory MISSING (CRITICAL)")

    if "meta" not in mf.get("missing_directories", []):
        for fname in EXPECTED_META_FILES:
            st = mf["meta_files_status"].get(fname, {})
            if not st.get("exists"):
                critical = " (CRITICAL)" if fname in CRITICAL_META_FILES else ""
                print(f"[MISSING-FILES] {name}: meta/{fname} MISSING{critical}")
            elif st.get("is_empty"):
                critical = " (CRITICAL)" if fname in CRITICAL_META_FILES else ""
                print(f"[MISSING-FILES] {name}: meta/{fname} EMPTY (0 bytes){critical}")

    if "data" in mf.get("missing_directories", []):
        print(f"[MISSING-FILES] {name}: parquet checks skipped (data/ missing)")
    elif mf["empty_parquet_files"]:
        head = mf["empty_parquet_files"][:10]
        more = "" if len(mf["empty_parquet_files"]) <= 10 else f" ... (+{len(mf['empty_parquet_files']) - 10} more)"
        print(f"[MISSING-FILES] {name}: empty parquet files: {head}{more}")

    if "data" not in mf.get("missing_directories", []):
        def _show(label, lst):
            if not lst:
                return
            head = lst[:20]
            more = "" if len(lst) <= 20 else f" ... (+{len(lst) - 20} more)"
            print(f"[MISSING-FILES] {name}: {label}: {head}{more}")

        _show("missing parquet by index gap", mf["missing_parquet_indices_by_gap"])
        _show("declared in episodes.jsonl but parquet missing", mf["missing_parquet_from_episodes_jsonl"])
        _show("declared in episodes_stats.jsonl but parquet missing", mf["missing_parquet_from_episodes_stats_jsonl"])
        _show("declared by info.json total_episodes but parquet missing", mf["missing_parquet_from_declared_total"])


# ============================================================
# 以下沿用原脚本，只在 process_one_id 中加入 missing_files_check 字段
# ============================================================

def process_one_id(id_root):
    data_dir = id_root / "data"
    meta_dir = id_root / "meta"

    meta_report = summarize_meta(meta_dir)
    episode_files = find_episode_files(data_dir) if data_dir.exists() else []

    reports = []
    for ep in episode_files:
        reports.append(summarize_episode(ep, meta_report, dataset_root=id_root))

    summary = build_summary(reports)

    num_parquet = len(episode_files)
    num_eps = meta_report.get("num_episodes_jsonl")
    num_eps_stats = meta_report.get("num_episodes_stats_jsonl")
    meta_vs_data = {
        "num_parquet": num_parquet,
        "num_episodes_jsonl": num_eps,
        "num_episodes_stats_jsonl": num_eps_stats,
        "episodes_jsonl_missing": (num_parquet - num_eps) if num_eps is not None else None,
        "episodes_stats_jsonl_missing": (num_parquet - num_eps_stats) if num_eps_stats is not None else None,
        "episodes_jsonl_match": (num_eps == num_parquet) if num_eps is not None else False,
        "episodes_stats_jsonl_match": (num_eps_stats == num_parquet) if num_eps_stats is not None else False,
    }

    indices = []
    unparsed_files = []
    for ep in episode_files:
        idx = extract_episode_index(ep)
        if idx is None:
            unparsed_files.append(str(ep))
        else:
            indices.append(idx)

    found_set = set(indices)
    duplicate_indices = sorted({i for i in indices if indices.count(i) > 1})

    declared_total = None
    info = meta_report.get("info") or {}
    for k in ("total_episodes", "num_episodes"):
        v = info.get(k)
        if isinstance(v, int):
            declared_total = v
            break

    if found_set:
        expected_end = max(max(found_set) + 1, declared_total or 0)
        missing_indices = sorted(set(range(expected_end)) - found_set)
    else:
        missing_indices = []

    parquet_check = {
        "num_parquet": num_parquet,
        "num_parsed_indices": len(indices),
        "unparsed_files": unparsed_files,
        "min_index": min(found_set) if found_set else None,
        "max_index": max(found_set) if found_set else None,
        "declared_total_episodes": declared_total,
        "missing_indices": missing_indices,
        "num_missing": len(missing_indices),
        "duplicate_indices": duplicate_indices,
        "is_continuous": (len(missing_indices) == 0 and not unparsed_files),
    }

    parquet_idx_to_rows = {}
    for ep, rep in zip(episode_files, reports):
        idx = extract_episode_index(ep)
        if idx is None or not rep.get("load_ok"):
            continue
        parquet_idx_to_rows[idx] = rep.get("num_rows")

    idx_to_len = meta_report.get("episodes_index_to_length") or {}
    episodes_jsonl_idx_set = set(idx_to_len.keys())
    episodes_stats_idx_set = set(meta_report.get("episodes_stats_episode_indices") or [])

    index_consistency = {
        "parquet_minus_episodes_jsonl": sorted(found_set - episodes_jsonl_idx_set),
        "episodes_jsonl_minus_parquet": sorted(episodes_jsonl_idx_set - found_set),
        "episodes_jsonl_index_set_match": found_set == episodes_jsonl_idx_set,
        "parquet_minus_episodes_stats_jsonl": sorted(found_set - episodes_stats_idx_set),
        "episodes_stats_jsonl_minus_parquet": sorted(episodes_stats_idx_set - found_set),
        "episodes_stats_jsonl_index_set_match": found_set == episodes_stats_idx_set,
    }

    length_mismatches = []
    idx_missing_in_jsonl = []
    idx_with_null_length = []
    for idx, pq_rows in sorted(parquet_idx_to_rows.items()):
        if idx not in idx_to_len:
            idx_missing_in_jsonl.append(idx)
            continue
        jsonl_len = idx_to_len[idx]
        if jsonl_len is None:
            idx_with_null_length.append(idx)
            continue
        if jsonl_len != pq_rows:
            length_mismatches.append({
                "episode_index": idx,
                "jsonl_length": jsonl_len,
                "parquet_length": pq_rows,
            })

    length_check = {
        "checked_episodes": len(parquet_idx_to_rows),
        "num_length_mismatches": len(length_mismatches),
        "num_idx_missing_in_jsonl": len(idx_missing_in_jsonl),
        "num_idx_with_null_length": len(idx_with_null_length),
        "length_mismatches": length_mismatches,
        "idx_missing_in_jsonl": idx_missing_in_jsonl,
        "idx_with_null_length": idx_with_null_length,
        "all_lengths_match": (
            len(length_mismatches) == 0
            and len(idx_missing_in_jsonl) == 0
        ),
    }

    parquet_size_check = check_parquet_sizes(episode_files)

    # NEW: 缺失文件诊断
    missing_files_check = diagnose_missing_files(id_root, episode_files, meta_report)

    return {
        "id_root": str(id_root),
        "meta_report": meta_report,
        "episode_files": [str(x) for x in episode_files],
        "episode_reports": reports,
        "summary": summary,
        "meta_vs_data": meta_vs_data,
        "parquet_check": parquet_check,
        "parquet_size_check": parquet_size_check,
        "index_consistency": index_consistency,
        "length_check": length_check,
        "missing_files_check": missing_files_check,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/public/datasets/franka_tele_data/rank_1")
    parser.add_argument("--out_dir", type=str, default="franka_inspect_outputs")
    parser.add_argument("--num_vis", type=int, default=5, help="随机导出的视频总数，不是每个 id 的数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_video", dest="save_video", action="store_true", default=True,
                        help="是否生成 MP4 视频（默认 True）")
    parser.add_argument("--no_video", dest="save_video", action="store_false",
                        help="禁用 MP4 视频生成")
    parser.add_argument("--save_last_frame", action="store_true", default=False,
                        help="是否额外保存最后一帧图片")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_roots = discover_id_roots(root)
    print(f"[INFO] found {len(id_roots)} id roots under {root}")

    if len(id_roots) == 0:
        print(f"[ERROR] no valid id directories found under {root}")
        return

    all_id_reports = []
    all_episode_candidates = []

    for id_root in id_roots:
        print(f"[INFO] processing id root: {id_root}")
        id_result = process_one_id(id_root)
        all_id_reports.append(id_result)

        print(f"[INFO] {id_root.name}: found {len(id_result['episode_files'])} parquet episodes")
        print(f"[INFO] {id_root.name}: image feature candidates from meta: {id_result['meta_report'].get('image_feature_candidates', [])}")

        # NEW: 缺失文件诊断打印
        print_missing_files_report(id_root, id_result["missing_files_check"])

        mvd = id_result["meta_vs_data"]
        tag_eps = "OK" if mvd["episodes_jsonl_match"] else "MISMATCH"
        tag_stats = "OK" if mvd["episodes_stats_jsonl_match"] else "MISMATCH"
        print(
            f"[META-CHECK] {id_root.name}: parquet={mvd['num_parquet']} "
            f"episodes.jsonl={mvd['num_episodes_jsonl']} [{tag_eps}] "
            f"episodes_stats.jsonl={mvd['num_episodes_stats_jsonl']} [{tag_stats}]"
        )
        if not mvd["episodes_jsonl_match"]:
            print(f"[WARN] {id_root.name}: episodes.jsonl missing {mvd['episodes_jsonl_missing']} entries vs parquet")

        pc = id_result["parquet_check"]
        tag_pq = "OK" if pc["is_continuous"] else "GAPS"
        print(
            f"[PARQUET-CHECK] {id_root.name}: count={pc['num_parquet']} "
            f"range=[{pc['min_index']}..{pc['max_index']}] "
            f"declared_total={pc['declared_total_episodes']} "
            f"missing={pc['num_missing']} [{tag_pq}]"
        )
        if pc["missing_indices"]:
            preview = pc["missing_indices"][:20]
            more = "" if len(pc["missing_indices"]) <= 20 else f" ... (+{len(pc['missing_indices']) - 20} more)"
            print(f"[WARN] {id_root.name}: missing episode_index: {preview}{more}")
        if pc["duplicate_indices"]:
            print(f"[WARN] {id_root.name}: duplicate episode_index: {pc['duplicate_indices']}")
        if pc["unparsed_files"]:
            print(f"[WARN] {id_root.name}: {len(pc['unparsed_files'])} parquet files have no episode_NNNNNN.parquet pattern")

        ic = id_result["index_consistency"]
        tag_eps_idx = "OK" if ic["episodes_jsonl_index_set_match"] else "DIFF"
        tag_stats_idx = "OK" if ic["episodes_stats_jsonl_index_set_match"] else "DIFF"
        print(
            f"[INDEX-CHECK] {id_root.name}: episodes.jsonl [{tag_eps_idx}] "
            f"episodes_stats.jsonl [{tag_stats_idx}]"
        )

        def _preview(name, lst):
            if not lst:
                return
            head = lst[:20]
            more = "" if len(lst) <= 20 else f" ... (+{len(lst) - 20} more)"
            print(f"[WARN] {id_root.name}: {name}: {head}{more}")

        _preview("in parquet but NOT in episodes.jsonl", ic["parquet_minus_episodes_jsonl"])
        _preview("in episodes.jsonl but NOT in parquet", ic["episodes_jsonl_minus_parquet"])
        _preview("in parquet but NOT in episodes_stats.jsonl", ic["parquet_minus_episodes_stats_jsonl"])
        _preview("in episodes_stats.jsonl but NOT in parquet", ic["episodes_stats_jsonl_minus_parquet"])

        lc = id_result["length_check"]
        tag_len = "OK" if lc["all_lengths_match"] else "MISMATCH"
        print(
            f"[LENGTH-CHECK] {id_root.name}: checked={lc['checked_episodes']} "
            f"mismatches={lc['num_length_mismatches']} "
            f"idx_missing_in_jsonl={lc['num_idx_missing_in_jsonl']} "
            f"null_length={lc['num_idx_with_null_length']} [{tag_len}]"
        )
        if lc["length_mismatches"]:
            head = lc["length_mismatches"][:10]
            more = "" if len(lc["length_mismatches"]) <= 10 else f" ... (+{len(lc['length_mismatches']) - 10} more)"
            print(f"[WARN] {id_root.name}: length mismatches (jsonl vs parquet rows): {head}{more}")
        if not mvd["episodes_stats_jsonl_match"]:
            print(f"[WARN] {id_root.name}: episodes_stats.jsonl missing {mvd['episodes_stats_jsonl_missing']} entries vs parquet")

        psc = id_result["parquet_size_check"]
        if psc.get("error"):
            print(f"[SIZE-CHECK] {id_root.name}: {psc['error']}")
        else:
            n_small = psc.get("num_too_small", 0)
            n_large = psc.get("num_too_large", 0)
            if psc["num_anomalous"] == 0:
                tag_anomaly = "OK"
            else:
                parts = []
                if n_small:
                    parts.append(f"过小={n_small}")
                if n_large:
                    parts.append(f"过大={n_large}")
                tag_anomaly = f"ANOMALY({', '.join(parts)})"
            print(
                f"[SIZE-CHECK] {id_root.name}: files={psc['num_files']} "
                f"mean={psc['mean_bytes']:.0f} bytes "
                f"std={psc['std_bytes']:.0f} "
                f"z_threshold={psc.get('z_score_threshold', PARQUET_SIZE_Z_THRESHOLD)} "
                f"[{tag_anomaly}]"
            )
            if psc.get("note"):
                print(f"[SIZE-CHECK] {id_root.name}: {psc['note']}")

            def _print_size_issues(label, items, max_show=10):
                if not items:
                    return
                for item in items[:max_show]:
                    print(
                        f"[WARN] {id_root.name}: parquet {label} "
                        f"{item['file']}  size={item['size_bytes']} bytes  "
                        f"z={item['z_score']}"
                    )
                if len(items) > max_show:
                    print(
                        f"[WARN] {id_root.name}: parquet {label} "
                        f"... (+{len(items) - max_show} more)"
                    )

            _print_size_issues("文件过小", psc.get("too_small_files", []))
            _print_size_issues("文件过大", psc.get("too_large_files", []))

        for ep in id_result["episode_files"]:
            all_episode_candidates.append({
                "id_root": str(id_root),
                "episode_file": ep,
            })

    global_episode_reports = []
    for id_result in all_id_reports:
        global_episode_reports.extend(id_result["episode_reports"])

    global_summary = build_summary(global_episode_reports)

    rng = random.Random(args.seed)

    size_anomaly_map = {}
    quality_anomaly_map = {}
    for r in all_id_reports:
        anomalous_names = {x["file"] for x in r["parquet_size_check"].get("anomalous_files", [])}
        size_anomaly_map[r["id_root"]] = anomalous_names

        anomalous_paths = {er["file"] for er in r["episode_reports"] if er.get("issues")}
        quality_anomaly_map[r["id_root"]] = anomalous_paths

    anomalous_candidates = []
    normal_candidates = []

    for item in all_episode_candidates:
        id_root_str = item["id_root"]
        ep_path = item["episode_file"]
        ep = Path(ep_path)

        is_size_anomaly = ep.name in size_anomaly_map.get(id_root_str, set())
        is_quality_anomaly = ep_path in quality_anomaly_map.get(id_root_str, set())

        if is_size_anomaly or is_quality_anomaly:
            anomalous_candidates.append(item)
        else:
            normal_candidates.append(item)

    chosen = anomalous_candidates.copy()
    num_normal_to_sample = min(args.num_vis, len(normal_candidates))
    if num_normal_to_sample > 0:
        chosen.extend(rng.sample(normal_candidates, num_normal_to_sample))

    video_reports = []
    for i, item in enumerate(chosen):
        id_root = Path(item["id_root"])
        ep = Path(item["episode_file"])

        id_meta_report = None
        for x in all_id_reports:
            if x["id_root"] == str(id_root):
                id_meta_report = x["meta_report"]
                break

        id_out_dir = out_dir / id_root.name
        id_out_dir.mkdir(parents=True, exist_ok=True)
        video_path = id_out_dir / f"sample_{i:02d}_{ep.stem}.mp4"

        print(f"[INFO] exporting video for {ep}")
        report = export_episode_video(ep, video_path, id_meta_report, dataset_root=id_root,
                                       save_video=args.save_video, save_last_frame=True)
        report["id_root"] = str(id_root)
        video_reports.append(report)

    meta_check_global = {
        "ids_with_episodes_jsonl_mismatch": [
            Path(r["id_root"]).name
            for r in all_id_reports
            if not r["meta_vs_data"]["episodes_jsonl_match"]
        ],
        "ids_with_episodes_stats_jsonl_mismatch": [
            Path(r["id_root"]).name
            for r in all_id_reports
            if not r["meta_vs_data"]["episodes_stats_jsonl_match"]
        ],
        "ids_with_parquet_gaps": [
            {
                "id": Path(r["id_root"]).name,
                "num_missing": r["parquet_check"]["num_missing"],
                "missing_indices": r["parquet_check"]["missing_indices"][:50],
                "duplicate_indices": r["parquet_check"]["duplicate_indices"],
            }
            for r in all_id_reports
            if not r["parquet_check"]["is_continuous"]
        ],
        "ids_with_index_set_diff": [
            {
                "id": Path(r["id_root"]).name,
                "parquet_minus_episodes_jsonl": r["index_consistency"]["parquet_minus_episodes_jsonl"][:50],
                "episodes_jsonl_minus_parquet": r["index_consistency"]["episodes_jsonl_minus_parquet"][:50],
                "parquet_minus_episodes_stats_jsonl": r["index_consistency"]["parquet_minus_episodes_stats_jsonl"][:50],
                "episodes_stats_jsonl_minus_parquet": r["index_consistency"]["episodes_stats_jsonl_minus_parquet"][:50],
            }
            for r in all_id_reports
            if not (
                r["index_consistency"]["episodes_jsonl_index_set_match"]
                and r["index_consistency"]["episodes_stats_jsonl_index_set_match"]
            )
        ],
        "ids_with_length_mismatch": [
            {
                "id": Path(r["id_root"]).name,
                "num_length_mismatches": r["length_check"]["num_length_mismatches"],
                "num_idx_missing_in_jsonl": r["length_check"]["num_idx_missing_in_jsonl"],
                "length_mismatches": r["length_check"]["length_mismatches"][:50],
            }
            for r in all_id_reports
            if not r["length_check"]["all_lengths_match"]
        ],
        "ids_with_parquet_size_anomalies": [
            {
                "id": Path(r["id_root"]).name,
                "num_anomalous": r["parquet_size_check"]["num_anomalous"],
                "num_too_small": r["parquet_size_check"].get("num_too_small", 0),
                "num_too_large": r["parquet_size_check"].get("num_too_large", 0),
                "z_score_threshold": r["parquet_size_check"].get("z_score_threshold"),
                "too_small_files": r["parquet_size_check"].get("too_small_files", [])[:50],
                "too_large_files": r["parquet_size_check"].get("too_large_files", [])[:50],
                "anomalous_files": r["parquet_size_check"]["anomalous_files"][:50],
            }
            for r in all_id_reports
            if r["parquet_size_check"].get("num_anomalous", 0) > 0
        ],
    }
    global_summary["meta_check"] = meta_check_global

    # NEW: 全局缺失文件汇总
    missing_files_global = {
        "ids_with_missing_directories": [
            {
                "id": Path(r["id_root"]).name,
                "missing_directories": r["missing_files_check"]["missing_directories"],
            }
            for r in all_id_reports
            if r["missing_files_check"]["missing_directories"]
        ],
        "ids_with_missing_data_dir": [
            Path(r["id_root"]).name
            for r in all_id_reports
            if "data" in r["missing_files_check"]["missing_directories"]
        ],
        "ids_with_missing_meta_dir": [
            Path(r["id_root"]).name
            for r in all_id_reports
            if "meta" in r["missing_files_check"]["missing_directories"]
        ],
        "ids_with_missing_meta_files": [
            {
                "id": Path(r["id_root"]).name,
                "missing_meta_files": r["missing_files_check"]["missing_meta_files"],
                "empty_meta_files": r["missing_files_check"]["empty_meta_files"],
                "missing_critical_meta_files": r["missing_files_check"]["missing_critical_meta_files"],
            }
            for r in all_id_reports
            if (
                r["missing_files_check"]["missing_meta_files"]
                or r["missing_files_check"]["empty_meta_files"]
            )
        ],
        "ids_with_missing_or_empty_parquet": [
            {
                "id": Path(r["id_root"]).name,
                "empty_parquet_files": r["missing_files_check"]["empty_parquet_files"],
                "num_missing_parquet_indices": len(
                    r["missing_files_check"]["missing_parquet_indices_union"]
                ),
                "missing_parquet_indices_union": r["missing_files_check"][
                    "missing_parquet_indices_union"
                ][:100],
                "missing_parquet_from_episodes_jsonl": r["missing_files_check"][
                    "missing_parquet_from_episodes_jsonl"
                ][:100],
                "missing_parquet_from_episodes_stats_jsonl": r["missing_files_check"][
                    "missing_parquet_from_episodes_stats_jsonl"
                ][:100],
                "missing_parquet_from_declared_total": r["missing_files_check"][
                    "missing_parquet_from_declared_total"
                ][:100],
            }
            for r in all_id_reports
            if (
                r["missing_files_check"]["empty_parquet_files"]
                or r["missing_files_check"]["missing_parquet_indices_union"]
            )
        ],
        "ids_all_ok": [
            Path(r["id_root"]).name
            for r in all_id_reports
            if r["missing_files_check"]["is_ok"]
        ],
    }
    global_summary["missing_files_check"] = missing_files_global

    output = {
        "scan_root": str(root),
        "num_id_roots": len(id_roots),
        "global_summary": global_summary,
        "per_id_reports": all_id_reports,
        "video_reports": video_reports,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("[GLOBAL SUMMARY]")
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))
    print(f"[INFO] outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
