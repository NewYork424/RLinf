#!/usr/bin/env python3
"""
State-Image 对齐检查 - 支持 PNG 编码的图像

Usage:
    python check_alignment_png.py path/to/deployment_data
"""

import pandas as pd
import numpy as np
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io


def decode_png_image(image_dict):
    """从 PNG 二进制数据解码图像"""
    if isinstance(image_dict, dict):
        # 图像以 'data' 键存储为 PNG 字节
        if 'data' in image_dict:
            png_bytes = image_dict['data']
        else:
            # 或者直接取第一个值
            png_bytes = next(iter(image_dict.values()))

        # 解码 PNG
        if isinstance(png_bytes, bytes):
            img = Image.open(io.BytesIO(png_bytes))
            return np.array(img)

    return None


def check_alignment(data_dir):
    """显示关键帧的图像和对应的 state 值"""

    data_path = Path(data_dir)
    files = sorted(glob.glob(f'{data_dir}/rank_0/id_0/data/chunk-000/episode_*.parquet'))

    if not files:
        print(f"❌ No data found in {data_dir}")
        return

    df = pd.read_parquet(files[0])
    ep_name = files[0].split('/')[-1]

    print(f"\n{'='*80}")
    print(f"State-Image Alignment Check (PNG Format)")
    print(f"Episode: {ep_name}")
    print(f"Total frames: {len(df)}")
    print(f"{'='*80}\n")

    # 选择关键帧：开始、1/4、中间、3/4、结束
    total_frames = len(df)
    key_indices = [
        0,
        total_frames // 4,
        total_frames // 2,
        3 * total_frames // 4,
        total_frames - 1
    ]

    print("Decoding PNG images and extracting state values...\n")

    frame_data_list = []

    for idx in key_indices:
        image_dict = df['image'].iloc[idx]
        state = df['state'].iloc[idx]

        # 解码 PNG
        image_array = decode_png_image(image_dict)
        state_array = np.array(state)

        if image_array is not None:
            print(f"  ✓ Frame {idx}: Image decoded, shape={image_array.shape}, State shape={state_array.shape}")
        else:
            print(f"  ✗ Frame {idx}: Failed to decode image")

        frame_data_list.append({
            'idx': idx,
            'image': image_array,
            'state': state_array
        })

    print("\n创建可视化...\n")

    # 创建可视化
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f'State-Image Alignment Check\n{ep_name}\n(Top: Robot Images | Bottom: State Values)',
                fontsize=16, fontweight='bold')

    for plot_idx, frame_data in enumerate(frame_data_list):
        frame_idx = frame_data['idx']
        image_array = frame_data['image']
        state_array = frame_data['state']

        # ===== 第一行：显示图像 =====
        ax_img = plt.subplot(2, 5, plot_idx + 1)

        if image_array is not None:
            try:
                ax_img.imshow(image_array)
                ax_img.set_title(f'Frame {frame_idx}', fontweight='bold', fontsize=12)
            except Exception as e:
                ax_img.text(0.5, 0.5, f'Display error:\n{str(e)[:30]}',
                           ha='center', va='center', transform=ax_img.transAxes,
                           fontsize=10, color='red')
                ax_img.set_title(f'Frame {frame_idx} (Error)', fontweight='bold', color='red')
        else:
            ax_img.text(0.5, 0.5, 'Failed to decode PNG',
                       ha='center', va='center', transform=ax_img.transAxes,
                       fontsize=12, color='red', fontweight='bold')
            ax_img.set_title(f'Frame {frame_idx} (Failed)', fontweight='bold', color='red')

        ax_img.axis('off')

        # ===== 第二行：显示 state 值 =====
        ax_state = plt.subplot(2, 5, plot_idx + 6)

        # 构建 state 文本
        state_text = f"Frame #{frame_idx}\n"
        state_text += f"{'─'*18}\n"
        state_text += f"Shape: {state_array.shape}\n"
        state_text += f"Min: {state_array.min():+.4f}\n"
        state_text += f"Max: {state_array.max():+.4f}\n"
        state_text += f"Mean: {state_array.mean():+.4f}\n"
        state_text += f"Std: {state_array.std():.4f}\n\n"
        state_text += f"Values (all {len(state_array)} dims):\n"

        # 显示所有 state 值（因为只有 20 维）
        for i in range(len(state_array)):
            state_text += f"  [{i:2d}]: {state_array[i]:+.4f}\n"

        ax_state.text(0.02, 0.98, state_text, verticalalignment='top',
                     fontsize=8, family='monospace', transform=ax_state.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))
        ax_state.axis('off')

    plt.tight_layout()

    # 保存
    output_file = 'state_image_alignment_check.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}\n")

    # ===== 打印检查清单 =====
    print(f"{'='*80}")
    print("📋 Manual Inspection Checklist:")
    print(f"{'='*80}\n")

    print("Open the file: state_image_alignment_check.png\n")

    print("For each of the 5 columns:")
    print("  TOP ROW:    Robot image at that moment in time")
    print("  BOTTOM ROW: Corresponding state values (20 dimensions)\n")

    print("✅ CORRECT ALIGNMENT means:")
    print("  • Robot position changes left to right in images")
    print("  • State values also change correspondingly")
    print("  • E.g., if robot moves up → some state dims should increase")
    print("  • If robot moves left → different dims should change\n")

    print("❌ INCORRECT ALIGNMENT means:")
    print("  • Images change but state values stay the same")
    print("  • State values all zeros or constant\n")

    print("🎯 Key Questions:")
    print("  Q1: Do robot positions change across frames? YES/NO")
    print("  Q2: Do state values change across frames? YES/NO")
    print("  Q3: When robot moves, do state values respond? YES/NO\n")

    print("✅ If ALL answers are YES → Alignment is CORRECT")
    print("❌ If ANY answer is NO → There might be an issue\n")

    print(f"{'='*80}\n")

    # 也显示在屏幕上
    try:
        plt.show()
        print("Plot displayed successfully!")
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print("But the image was saved to disk.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        print("Usage: python check_alignment_png.py path/to/deployment_data")
        sys.exit(1)

    check_alignment(data_dir)
