#!/usr/bin/env python3
"""
Visualize Atari data from the shard
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io

print(">>> Visualizing Atari data")

# ---- Setup ----
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"

# ---- PNG-based parser (from fixed version) ----
def _decode_obs_png(var_sparse):
    """Decode PNG-encoded observations; returns [84,84,4] uint8."""
    var_dense = tf.sparse.to_dense(var_sparse)
    
    def decode_png(png_bytes):
        png_decoded = tf.io.decode_png(png_bytes, channels=1)
        return tf.cast(png_decoded, tf.uint8)
    
    decoded_frames = tf.map_fn(
        decode_png,
        var_dense,
        fn_output_signature=tf.uint8,
    )  # [4, 84, 84, 1]
    
    stacked = tf.transpose(decoded_frames, [1, 2, 0, 3])
    return tf.squeeze(stacked, axis=3)  # [84, 84, 4]

def parse_transition_png(rec):
    features = {
        "o_t":   tf.io.VarLenFeature(tf.string),
        "a_t":   tf.io.FixedLenFeature([], tf.int64),
        "r_t":   tf.io.FixedLenFeature([], tf.float32),
        "d_t":   tf.io.FixedLenFeature([], tf.float32),
        "o_tp1": tf.io.VarLenFeature(tf.string),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
    }
    p = tf.io.parse_single_example(rec, features)
    o_t   = _decode_obs_png(p["o_t"])
    o_tp1 = _decode_obs_png(p["o_tp1"])
    a_t   = tf.cast(p["a_t"], tf.int32)
    r_tp1 = p["r_t"]
    done  = tf.cast(tf.equal(p["d_t"], 0.0), tf.float32)
    return o_t, a_t, r_tp1, o_tp1, done

# ---- Load data ----
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Get a few samples
samples = list(ds.take(3))
print(f">>> Loaded {len(samples)} samples")

# ---- Visualization functions ----
def plot_observation(obs, title="Observation", figsize=(12, 3)):
    """Plot the 4-frame observation"""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    for i in range(4):
        frame = obs[:, :, i]
        axes[i].imshow(frame, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_transition(obs_t, obs_tp1, action, reward, done, figsize=(15, 6)):
    """Plot a complete transition (o_t -> action -> o_tp1)"""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(f'Transition: action={action}, reward={reward:.2f}, done={bool(done)}', fontsize=14)
    
    # Current observation (o_t)
    for i in range(4):
        frame = obs_t[:, :, i]
        axes[0, i].imshow(frame, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'o_t Frame {i+1}')
        axes[0, i].axis('off')
    
    # Next observation (o_tp1)
    for i in range(4):
        frame = obs_tp1[:, :, i]
        axes[1, i].imshow(frame, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'o_tp1 Frame {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_frame_differences(obs_t, obs_tp1, figsize=(12, 4)):
    """Plot differences between consecutive frames to see motion"""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle('Frame Differences (motion detection)', fontsize=14)
    
    for i in range(4):
        diff = obs_tp1[:, :, i].astype(np.float32) - obs_t[:, :, i].astype(np.float32)
        im = axes[i].imshow(diff, cmap='RdBu', vmin=-50, vmax=50)
        axes[i].set_title(f'Frame {i+1} Diff')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

# ---- Visualize samples ----
for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(samples):
    print(f"\n>>> Sample {i+1}:")
    print(f"  Action: {a_t.numpy()}")
    print(f"  Reward: {r_tp1.numpy():.2f}")
    print(f"  Done: {bool(done.numpy())}")
    print(f"  Observation shapes: o_t={o_t.shape}, o_tp1={o_tp1.shape}")
    
    # Convert to numpy
    o_t_np = o_t.numpy()
    o_tp1_np = o_tp1.numpy()
    action = a_t.numpy()
    reward = r_tp1.numpy()
    done_flag = done.numpy()
    
    # Plot current observation
    fig1 = plot_observation(o_t_np, f"Sample {i+1} - Current Observation (o_t)")
    plt.show()
    
    # Plot next observation
    fig2 = plot_observation(o_tp1_np, f"Sample {i+1} - Next Observation (o_tp1)")
    plt.show()
    
    # Plot transition
    fig3 = plot_transition(o_t_np, o_tp1_np, action, reward, done_flag)
    plt.show()
    
    # Plot frame differences
    fig4 = plot_frame_differences(o_t_np, o_tp1_np)
    plt.show()
    
    # Show some statistics
    print(f"  o_t value range: {o_t_np.min()} - {o_t_np.max()}")
    print(f"  o_tp1 value range: {o_tp1_np.min()} - {o_tp1_np.max()}")
    print(f"  o_t mean: {o_t_np.mean():.2f}")
    print(f"  o_tp1 mean: {o_tp1_np.mean():.2f}")

print("\n>>> Visualization complete!")
print(">>> Tips:")
print("  - Each observation has 4 consecutive frames")
print("  - Frames are grayscale (0-255)")
print("  - Frame differences show motion/change")
print("  - This is Breakout - look for the paddle and ball!")
