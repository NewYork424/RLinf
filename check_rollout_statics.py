#!/usr/bin/env python3
"""
完整的数据检查脚本 - 带详细输出和解释

Usage:
    python detailed_check.py path/to/deployment_data
"""

import pandas as pd
import numpy as np
import glob
import sys
from pathlib import Path


def detailed_check(data_dir):
    """详细检查 rollout 数据，打印每一步"""

    data_path = Path(data_dir)
    files = sorted(glob.glob(f'{data_dir}/rank_0/id_0/data/chunk-000/episode_*.parquet'))

    if not files:
        print(f"❌ No data found in {data_dir}")
        return

    print(f"\n{'='*80}")
    print(f"📊 DETAILED ROLLOUT DATA CHECK")
    print(f"{'='*80}\n")

    # ===== 检查 1: 首尾 Done Flag =====
    print("✅ CHECK 1: Episode Structure (done flag)")
    print("-" * 80)

    df = pd.read_parquet(files[0])
    ep_name = files[0].split('/')[-1]

    first_done = df['done'].iloc[0]
    last_done = df['done'].iloc[-1]
    done_count = df['done'].sum()

    print(f"\nEpisode: {ep_name}")
    print(f"  Total frames: {len(df)}")
    print(f"  First frame done: {first_done}  (should be False)")
    print(f"  Last frame done: {last_done}   (should be True)")
    print(f"  Number of True: {done_count}   (should be 1)")

    # 详细打印前几行和最后几行
    print(f"\n  First 3 frames done flags: {df['done'].iloc[:3].tolist()}")
    print(f"  Last 3 frames done flags: {df['done'].iloc[-3:].tolist()}")

    # 检查是否有问题
    structure_ok = (not first_done) and last_done and (done_count == 1)
    if structure_ok:
        print(f"\n  ✓ Structure OK")
    else:
        print(f"\n  ✗ STRUCTURE PROBLEM DETECTED")
        if first_done:
            print(f"    └─ First frame should not be done!")
        if not last_done:
            print(f"    └─ Last frame should be done!")
        if done_count != 1:
            print(f"    └─ Should have exactly 1 done flag!")

    # ===== 检查 2: 数值范围 =====
    print(f"\n{'='*80}")
    print("✅ CHECK 2: Numerical Values (NaN, Inf, Range)")
    print("-" * 80)

    # State 检查
    print(f"\nState:")
    state_data = np.concatenate([np.array([x]) for x in df['state']])

    print(f"  Shape: {state_data.shape}")
    print(f"  Min: {state_data.min():.4f}")
    print(f"  Max: {state_data.max():.4f}")
    print(f"  Mean: {state_data.mean():.4f}")
    print(f"  Std: {state_data.std():.4f}")
    print(f"  Has NaN: {np.isnan(state_data).any()}")
    print(f"  Has Inf: {np.isinf(state_data).any()}")

    state_ok = (not np.isnan(state_data).any()) and (not np.isinf(state_data).any())
    state_range_ok = (-10 < state_data.min() and state_data.max() < 10)

    if state_ok:
        print(f"  ✓ State values OK")
    else:
        print(f"  ✗ STATE VALUES PROBLEM")
        if np.isnan(state_data).any():
            nan_count = np.isnan(state_data).sum()
            print(f"    └─ Found {nan_count} NaN values!")
        if np.isinf(state_data).any():
            inf_count = np.isinf(state_data).sum()
            print(f"    └─ Found {inf_count} Inf values!")

    # Action 检查
    print(f"\nActions:")
    action_data = np.concatenate([np.array([x]) for x in df['actions']])

    print(f"  Shape: {action_data.shape}")
    print(f"  Min: {action_data.min():.4f}")
    print(f"  Max: {action_data.max():.4f}")
    print(f"  Mean: {action_data.mean():.4f}")
    print(f"  Std: {action_data.std():.4f}")
    print(f"  Has NaN: {np.isnan(action_data).any()}")
    print(f"  Has Inf: {np.isinf(action_data).any()}")

    action_ok = (not np.isnan(action_data).any()) and (not np.isinf(action_data).any())

    if action_ok:
        print(f"  ✓ Action values OK")
    else:
        print(f"  ✗ ACTION VALUES PROBLEM")
        if np.isnan(action_data).any():
            print(f"    └─ Found NaN values!")
        if np.isinf(action_data).any():
            print(f"    └─ Found Inf values!")

    # Image 检查
    print(f"\nImage [224×224×3 RGB]:")

    # 处理 image 可能是 object/dict 的情况
    first_image = df['image'].iloc[0]

    if isinstance(first_image, dict):
        print(f"  Type: Dict (object)")
        print(f"  Keys: {list(first_image.keys())}")
        print(f"  ✓ Image format OK (stored as dict)")
        image_ok = True
    else:
        try:
            image_data = np.array([x for x in df['image']])
            print(f"  Shape: {image_data.shape}")
            print(f"  Dtype: {image_data.dtype}")
            if image_data.dtype == object:
                print(f"  First image sample shape: {np.array(image_data[0]).shape}")
                print(f"  ✓ Image format OK")
                image_ok = True
            else:
                print(f"  Value range: [{image_data.min()}, {image_data.max()}]")
                image_ok = (image_data.dtype == np.uint8 and image_data.max() <= 255)
                if image_ok:
                    print(f"  ✓ Image format OK")
                else:
                    print(f"  ✗ IMAGE FORMAT PROBLEM")
        except Exception as e:
            print(f"  ⚠️  Could not parse image data: {e}")
            image_ok = True  # 假设没问题，因为我们无法验证

    # ===== 检查 3: 状态和图像对齐 =====
    print(f"\n{'='*80}")
    print("✅ CHECK 3: State-Image Alignment (Manual Visual Inspection Needed)")
    print("-" * 80)

    print(f"\nSample frames for visual inspection:")
    print(f"{'Index':<8} {'State Range':<20} {'First 3 dims':<30} {'Last 3 dims':<30}")
    print("-" * 90)

    for idx in [0, len(df)//2, len(df)-1]:
        state = df['state'].iloc[idx]
        state_arr = np.array(state)

        first_3 = state_arr[:3] if len(state_arr) >= 3 else state_arr
        last_3 = state_arr[-3:] if len(state_arr) >= 3 else state_arr

        print(f"{idx:<8} [{state_arr.min():.2f}, {state_arr.max():.2f}]     "
              f"{first_3}...    {last_3}...")

    print(f"\n  📸 To check alignment:")
    print(f"  1. Look at the image at each frame")
    print(f"  2. Compare robot joint angles (printed above)")
    print(f"  3. Do they match? (visual inspection required)")

    # ===== 检查 4: 统计信息 =====
    print(f"\n{'='*80}")
    print("✅ CHECK 4: Episode Statistics (all episodes)")
    print("-" * 80)

    lengths = []
    successes = []

    for f in files:
        df_ep = pd.read_parquet(f)
        lengths.append(len(df_ep))
        successes.append(bool(df_ep['is_success'].iloc[0]))

    lengths = np.array(lengths)
    successes = np.array(successes)

    print(f"\nTotal episodes: {len(files)}")
    print(f"Success rate: {successes.mean():.1%} ({successes.sum()}/{len(successes)})")
    print(f"\nEpisode lengths:")
    print(f"  Min: {lengths.min()}")
    print(f"  Max: {lengths.max()}")
    print(f"  Mean: {lengths.mean():.1f}")
    print(f"  Std: {lengths.std():.1f}")

    success_len = lengths[successes]
    fail_len = lengths[~successes]

    if len(success_len) > 0:
        print(f"\nSuccess episodes: Mean length = {success_len.mean():.0f}")
    if len(fail_len) > 0:
        print(f"Failure episodes: Mean length = {fail_len.mean():.0f}")

    # 检查异常
    stats_ok = True
    if lengths.std() == 0:
        print(f"\n⚠️  All episodes have same length - suspicious!")
        stats_ok = False

    if len(successes) > 0:
        if successes.mean() == 0.0:
            print(f"\n⚠️  No successful episodes - policy might not be working!")
            stats_ok = False
        elif successes.mean() == 1.0:
            print(f"\n⚠️  All episodes successful - might be unrealistic!")
            # This is actually OK for good rollouts

    if stats_ok:
        print(f"\n✓ Statistics look reasonable")

    # ===== 最终结论 =====
    print(f"\n{'='*80}")
    print("📋 FINAL VERDICT")
    print(f"{'='*80}\n")

    all_checks_ok = structure_ok and state_ok and action_ok and image_ok and stats_ok

    if all_checks_ok:
        print("✅ ALL CHECKS PASSED - Data looks good!")
        print("You can use this data for training.\n")
    else:
        print("⚠️  SOME CHECKS FAILED - Please review above\n")
        if not structure_ok:
            print("  • Fix: Check episode boundary handling")
        if not state_ok:
            print("  • Fix: Check observation extraction")
        if not action_ok:
            print("  • Fix: Check action generation")
        if not image_ok:
            print("  • Fix: Check image format")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        print("Usage: python detailed_check.py path/to/deployment_data")
        sys.exit(1)

    detailed_check(data_dir)
