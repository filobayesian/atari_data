#!/usr/bin/env python3
"""
Generate image sequences showing sequences with positive rewards
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import imageio

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
    
    # Stop after finding enough sequences
    if len(reward_sequences) >= 10:
        break

print(f">>> Found {len(reward_sequences)} reward sequences")

# ---- Generate image sequences ----
def create_image_sequence(sequence, output_dir, sequence_id):
    """Create a sequence of images showing the 4-frame observations"""
    os.makedirs(f"{output_dir}/sequence_{sequence_id}", exist_ok=True)
    
    for i, step_data in enumerate(sequence):
        obs = step_data['o_t']  # Shape: (84, 84, 4)
        action = step_data['a_t']
        reward = step_data['r_tp1']
        step = step_data['step']
        
        # Create a 2x2 grid showing all 4 frames
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f'Step {step} | Action: {action} | Reward: {reward:.1f}', fontsize=14)
        
        for frame_idx in range(4):
            row, col = frame_idx // 2, frame_idx % 2
            axes[row, col].imshow(obs[:, :, frame_idx], cmap='gray', vmin=0, vmax=255)
            axes[row, col].set_title(f'Frame {frame_idx + 1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sequence_{sequence_id}/frame_{i:03d}.png", 
                   dpi=100, bbox_inches='tight')
        plt.close(fig)

def create_gif_from_sequence(sequence, output_path, fps=2):
    """Create a GIF from a sequence of observations"""
    images = []
    
    for step_data in sequence:
        obs = step_data['o_t']
        action = step_data['a_t']
        reward = step_data['r_tp1']
        step = step_data['step']
        
        # Create 2x2 grid
        combined = np.zeros((84*2, 84*2), dtype=np.uint8)
        combined[0:84, 0:84] = obs[:, :, 0]
        combined[0:84, 84:168] = obs[:, :, 1]
        combined[84:168, 0:84] = obs[:, :, 2]
        combined[84:168, 84:168] = obs[:, :, 3]
        
        # Convert to PIL Image
        img = Image.fromarray(combined, mode='L')
        # Convert to RGB for text overlay
        img_rgb = img.convert('RGB')
        
        # Add text (simple approach - just save the image)
        images.append(img_rgb)
    
    # Save as GIF
    if images:
        imageio.mimsave(output_path, images, fps=fps)

# ---- Generate clips ----
print(">>> Generating reward clips...")
os.makedirs("reward_clips", exist_ok=True)

# Select the first few reward sequences
num_clips = min(4, len(reward_sequences))
selected_sequences = reward_sequences[:num_clips]

for i, sequence in enumerate(selected_sequences):
    print(f"  Creating clip {i+1}/{num_clips} (length: {len(sequence)} steps)...")
    
    # Create image sequence
    create_image_sequence(sequence, "reward_clips", i+1)
    
    # Create GIF clip
    gif_path = f"reward_clips/reward_sequence_{i+1}.gif"
    create_gif_from_sequence(sequence, gif_path, fps=2)
    
    # Print sequence info
    rewards_in_sequence = [step['r_tp1'] for step in sequence]
    positive_rewards = [r for r in rewards_in_sequence if r > 0]
    print(f"    Steps: {sequence[0]['step']} to {sequence[-1]['step']}")
    print(f"    Total reward: {sum(rewards_in_sequence):.1f}")
    print(f"    Positive rewards: {len(positive_rewards)} at steps {[j for j, r in enumerate(rewards_in_sequence) if r > 0]}")

print(f"\n>>> Generated {num_clips} reward clips in 'reward_clips/' directory")
print(">>> Files created:")
print("  - reward_sequence_X.gif: Animated GIFs (2 FPS)")
print("  - sequence_X/frame_XXX.png: Individual frame images")

# ---- Create a summary visualization ----
print("\n>>> Creating summary visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Reward Sequences Summary - 4-Frame Observations', fontsize=16)

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
        ax.set_title(f'Sequence {i+1} - Reward Frame\nStep {sequence[reward_idx]["step"]}, Action {sequence[reward_idx]["a_t"]}')
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
print("   Open the GIF files to see what happens when rewards occur!")
print("   Each GIF shows 4 consecutive frames (temporal information) in a 2x2 grid")

