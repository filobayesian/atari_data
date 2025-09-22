#!/usr/bin/env python3
"""
Generate clips showing the context around positive reward moments
- Shows what happened BEFORE and AFTER each positive reward
- Creates temporal sequences, not just 4-frame observations
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import imageio

print(">>> Generating reward context clips from Atari dataset")

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

# ---- Load dataset and find reward moments ----
print(">>> Loading dataset and finding reward moments...")
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Load all data to find reward moments
print(">>> Loading all transitions...")
all_transitions = []
for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(ds):
    all_transitions.append({
        'step': i,
        'o_t': o_t.numpy(),
        'a_t': a_t.numpy(),
        'r_tp1': r_tp1.numpy(),
        'o_tp1': o_tp1.numpy(),
        'done': done.numpy()
    })
    if i % 50000 == 0:
        print(f"  Processed {i} transitions...")

print(f">>> Loaded {len(all_transitions)} transitions")

# Find reward moments
reward_steps = [i for i, trans in enumerate(all_transitions) if trans['r_tp1'] > 0]
print(f">>> Found {len(reward_steps)} positive reward moments")

# ---- Generate context clips around each reward ----
def create_context_clip(reward_step, context_before=10, context_after=10):
    """Create a clip showing context around a reward moment"""
    start_step = max(0, reward_step - context_before)
    end_step = min(len(all_transitions), reward_step + context_after + 1)
    
    clip_transitions = all_transitions[start_step:end_step]
    reward_idx_in_clip = reward_step - start_step
    
    return clip_transitions, reward_idx_in_clip

def create_temporal_gif(transitions, reward_idx, output_path, fps=3):
    """Create a GIF showing temporal sequence of observations"""
    images = []
    
    for i, trans in enumerate(transitions):
        # Get the observation (4-frame stack)
        obs = trans['o_t']  # Shape: (84, 84, 4)
        action = trans['a_t']
        reward = trans['r_tp1']
        step = trans['step']
        
        # Create a single frame by taking the most recent frame (frame 3)
        # This shows the temporal progression
        current_frame = obs[:, :, 3]  # Most recent frame
        
        # Convert to RGB for better visualization
        frame_rgb = np.stack([current_frame] * 3, axis=-1)
        
        # Add text overlay
        frame_with_text = frame_rgb.copy()
        
        # Add border if this is the reward frame
        if i == reward_idx:
            # Add red border for reward frame
            frame_with_text[0:2, :] = [255, 0, 0]  # Top border
            frame_with_text[-2:, :] = [255, 0, 0]  # Bottom border
            frame_with_text[:, 0:2] = [255, 0, 0]  # Left border
            frame_with_text[:, -2:] = [255, 0, 0]  # Right border
        
        # Convert to PIL Image
        img = Image.fromarray(frame_with_text.astype(np.uint8))
        
        # Add text using PIL
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Add text
        text = f"Step: {step} | Action: {action} | Reward: {reward:.1f}"
        if i == reward_idx:
            text += " ⭐ REWARD!"
        
        draw.text((5, 5), text, fill=(255, 255, 0), font=font)
        
        images.append(img)
    
    # Save as GIF
    if images:
        imageio.mimsave(output_path, images, fps=fps)

def create_4frame_gif(transitions, reward_idx, output_path, fps=2):
    """Create a GIF showing 4-frame observations in 2x2 grid"""
    images = []
    
    for i, trans in enumerate(transitions):
        obs = trans['o_t']  # Shape: (84, 84, 4)
        action = trans['a_t']
        reward = trans['r_tp1']
        step = trans['step']
        
        # Create 2x2 grid of the 4 frames
        combined = np.zeros((84*2, 84*2), dtype=np.uint8)
        combined[0:84, 0:84] = obs[:, :, 0]  # Frame 1
        combined[0:84, 84:168] = obs[:, :, 1]  # Frame 2
        combined[84:168, 0:84] = obs[:, :, 2]  # Frame 3
        combined[84:168, 84:168] = obs[:, :, 3]  # Frame 4
        
        # Add red border if this is the reward frame
        if i == reward_idx:
            combined[0:3, :] = 255  # Top border
            combined[-3:, :] = 255  # Bottom border
            combined[:, 0:3] = 255  # Left border
            combined[:, -3:] = 255  # Right border
        
        # Convert to RGB
        frame_rgb = np.stack([combined] * 3, axis=-1)
        img = Image.fromarray(frame_rgb.astype(np.uint8))
        
        # Add text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        text = f"Step: {step} | Action: {action} | Reward: {reward:.1f}"
        if i == reward_idx:
            text += " ⭐ REWARD!"
        
        draw.text((10, 10), text, fill=(255, 255, 0), font=font)
        
        images.append(img)
    
    # Save as GIF
    if images:
        imageio.mimsave(output_path, images, fps=fps)

# ---- Generate clips ----
print(">>> Generating reward context clips...")
os.makedirs("reward_context_clips", exist_ok=True)

# Select first few reward moments
num_clips = min(4, len(reward_steps))
selected_reward_steps = reward_steps[:num_clips]

for i, reward_step in enumerate(selected_reward_steps):
    print(f"  Creating context clip {i+1}/{num_clips} around reward at step {reward_step}...")
    
    # Create context clip
    transitions, reward_idx = create_context_clip(reward_step, context_before=8, context_after=8)
    
    # Create temporal GIF (single frame progression)
    temporal_gif_path = f"reward_context_clips/temporal_context_{i+1}.gif"
    create_temporal_gif(transitions, reward_idx, temporal_gif_path, fps=3)
    
    # Create 4-frame GIF (showing all 4 frames in 2x2 grid)
    fourframe_gif_path = f"reward_context_clips/fourframe_context_{i+1}.gif"
    create_4frame_gif(transitions, reward_idx, fourframe_gif_path, fps=2)
    
    # Print context info
    print(f"    Context: steps {transitions[0]['step']} to {transitions[-1]['step']}")
    print(f"    Reward occurs at step {transitions[reward_idx]['step']} (index {reward_idx} in clip)")
    print(f"    Actions before reward: {[t['a_t'] for t in transitions[:reward_idx]]}")
    print(f"    Actions after reward: {[t['a_t'] for t in transitions[reward_idx+1:]]}")

print(f"\n>>> Generated {num_clips} reward context clips in 'reward_context_clips/' directory")
print(">>> Files created:")
print("  - temporal_context_X.gif: Shows single frame progression over time")
print("  - fourframe_context_X.gif: Shows 4-frame observations in 2x2 grid")
print("  - Red border indicates the exact reward moment")
print("  - ⭐ REWARD! text appears at the reward frame")

# ---- Create summary visualization ----
print("\n>>> Creating summary visualization...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Reward Context Analysis - What Happens Before and After Rewards', fontsize=16)

for i, reward_step in enumerate(selected_reward_steps[:4]):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Get context around this reward
    transitions, reward_idx = create_context_clip(reward_step, context_before=4, context_after=4)
    
    # Show the reward frame and a few frames before/after
    reward_frame = transitions[reward_idx]['o_t'][:, :, 3]  # Most recent frame
    
    ax.imshow(reward_frame, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Reward at Step {reward_step}\nAction: {transitions[reward_idx]["a_t"]}')
    ax.axis('off')
    
    # Add context info
    before_actions = [t['a_t'] for t in transitions[:reward_idx]]
    after_actions = [t['a_t'] for t in transitions[reward_idx+1:]]
    ax.text(0.02, 0.98, f'Before: {before_actions[-3:]}\nAfter: {after_actions[:3]}', 
           transform=ax.transAxes, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('reward_context_clips/context_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("  ✅ Saved reward_context_clips/context_summary.png")
print("\n🎉 Reward context analysis complete!")
print("   Now you can see what the agent was doing BEFORE and AFTER each reward!")
