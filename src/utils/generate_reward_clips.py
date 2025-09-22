#!/usr/bin/env python3
"""
Generate video clips showing sequences with positive rewards
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

print(">>> Generating reward clips from Atari dataset")

# ---- Setup ----
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"

# ---- PNG-based parser ----
def _decode_obs_png(var_sparse):
    var_dense = tf.sparse.to_dense(var_sparse)
    
    def decode_png(png_bytes):
        png_decoded = tf.io.decode_png(png_bytes, channels=1)
        return tf.cast(png_decoded, tf.uint8)
    
    decoded_frames = tf.map_fn(
        decode_png,
        var_dense,
        fn_output_signature=tf.uint8,
    )
    
    stacked = tf.transpose(decoded_frames, [1, 2, 0, 3])
    return tf.squeeze(stacked, axis=3)

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

# ---- Load dataset and find reward sequences ----
print(">>> Loading dataset and finding reward sequences...")
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Find sequences with positive rewards
reward_sequences = []
clip_length = 20  # 20 steps = ~5 seconds at 4 FPS
current_sequence = []
sequence_start = 0

print(">>> Scanning for reward sequences...")
for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(ds):
    current_sequence.append({
        'step': i,
        'o_t': o_t.numpy(),
        'a_t': a_t.numpy(),
        'r_tp1': r_tp1.numpy(),
        'o_tp1': o_tp1.numpy(),
        'done': done.numpy()
    })
    
    # If we found a positive reward, save this sequence
    if r_tp1.numpy() > 0:
        # Take the last clip_length steps (or all if shorter)
        start_idx = max(0, len(current_sequence) - clip_length)
        reward_sequence = current_sequence[start_idx:]
        reward_sequences.append(reward_sequence)
        print(f"  Found reward sequence at step {i}, length: {len(reward_sequence)}")
    
    # Keep only the last clip_length steps to avoid memory issues
    if len(current_sequence) > clip_length:
        current_sequence = current_sequence[-clip_length:]
    
    if i % 50000 == 0:
        print(f"  Processed {i} steps, found {len(reward_sequences)} reward sequences...")

print(f">>> Found {len(reward_sequences)} reward sequences")

# ---- Generate video clips ----
def create_video_clip(sequence, output_path, fps=4):
    """Create a video clip from a sequence of observations"""
    if len(sequence) == 0:
        return
    
    # Get frame dimensions
    height, width = sequence[0]['o_t'].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))
    
    for step_data in sequence:
        # Get the 4-frame observation
        obs = step_data['o_t']  # Shape: (84, 84, 4)
        action = step_data['a_t']
        reward = step_data['r_tp1']
        
        # Create a 2x2 grid showing all 4 frames
        frame_grid = np.zeros((height * 2, width * 2), dtype=np.uint8)
        frame_grid[0:height, 0:width] = obs[:, :, 0]  # Frame 1
        frame_grid[0:height, width:width*2] = obs[:, :, 1]  # Frame 2
        frame_grid[height:height*2, 0:width] = obs[:, :, 2]  # Frame 3
        frame_grid[height:height*2, width:width*2] = obs[:, :, 3]  # Frame 4
        
        # Add text overlay
        frame_with_text = cv2.cvtColor(frame_grid, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame_with_text, f"Step: {step_data['step']}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_with_text, f"Action: {action}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_with_text, f"Reward: {reward:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame_with_text)
    
    out.release()

def create_gif_clip(sequence, output_path, fps=2):
    """Create a GIF clip from a sequence of observations"""
    if len(sequence) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Reward Sequence (Step {sequence[0]["step"]})', fontsize=14)
    
    def animate(frame_idx):
        if frame_idx >= len(sequence):
            return
        
        step_data = sequence[frame_idx]
        obs = step_data['o_t']
        action = step_data['a_t']
        reward = step_data['r_tp1']
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # Plot each frame
        for i in range(4):
            row, col = i // 2, i % 2
            axes[row, col].imshow(obs[:, :, i], cmap='gray', vmin=0, vmax=255)
            axes[row, col].set_title(f'Frame {i+1}')
            axes[row, col].axis('off')
        
        # Add info text
        fig.text(0.5, 0.02, f'Step: {step_data["step"]} | Action: {action} | Reward: {reward:.1f}', 
                ha='center', fontsize=10)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(sequence), interval=1000//fps, repeat=True)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)

# ---- Generate clips ----
print(">>> Generating video clips...")
os.makedirs("reward_clips", exist_ok=True)

# Select the first few reward sequences
num_clips = min(4, len(reward_sequences))
selected_sequences = reward_sequences[:num_clips]

for i, sequence in enumerate(selected_sequences):
    print(f"  Creating clip {i+1}/{num_clips} (length: {len(sequence)} steps)...")
    
    # Create video clip
    video_path = f"reward_clips/reward_sequence_{i+1}.mp4"
    create_video_clip(sequence, video_path, fps=4)
    
    # Create GIF clip
    gif_path = f"reward_clips/reward_sequence_{i+1}.gif"
    create_gif_clip(sequence, gif_path, fps=2)
    
    # Print sequence info
    rewards_in_sequence = [step['r_tp1'] for step in sequence]
    positive_rewards = [r for r in rewards_in_sequence if r > 0]
    print(f"    Steps: {sequence[0]['step']} to {sequence[-1]['step']}")
    print(f"    Total reward: {sum(rewards_in_sequence):.1f}")
    print(f"    Positive rewards: {len(positive_rewards)} at steps {[i for i, r in enumerate(rewards_in_sequence) if r > 0]}")

print(f"\n>>> Generated {num_clips} reward clips in 'reward_clips/' directory")
print(">>> Files created:")
print("  - reward_sequence_X.mp4: Video clips (4 FPS)")
print("  - reward_sequence_X.gif: Animated GIFs (2 FPS)")

# ---- Create a summary visualization ----
print("\n>>> Creating summary visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Reward Sequences Summary', fontsize=16)

for i, sequence in enumerate(selected_sequences[:4]):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Show the frame where reward occurred
    reward_frames = [j for j, step in enumerate(sequence) if step['r_tp1'] > 0]
    if reward_frames:
        reward_idx = reward_frames[0]  # First reward in sequence
        obs = sequence[reward_idx]['o_t']
        
        # Create 2x2 grid of frames
        combined = np.zeros((84*2, 84*2), dtype=np.uint8)
        combined[0:84, 0:84] = obs[:, :, 0]
        combined[0:84, 84:168] = obs[:, :, 1]
        combined[84:168, 0:84] = obs[:, :, 2]
        combined[84:168, 84:168] = obs[:, :, 3]
        
        ax.imshow(combined, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Sequence {i+1} - Reward Frame')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'Sequence {i+1}\nNo reward in clip', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
plt.savefig('reward_clips/summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("  ✅ Saved reward_clips/summary.png")
print("\n🎉 Reward clip generation complete!")
print("   Open the video files to see what happens when rewards occur!")

