#!/usr/bin/env python3
"""
Sample 150k consecutive frames from Atari dataset maximizing positive rewards
"""
import os
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import time

print(">>> Sampling 150k consecutive frames with maximum positive rewards")

# ---- Setup ----
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"
TARGET_LENGTH = 150000  # 150k frames
WINDOW_SIZE = 10000  # Process in chunks of 10k for memory efficiency

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

# ---- Load dataset and analyze rewards ----
print(">>> Loading dataset and analyzing reward distribution...")
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# First pass: collect all rewards to find optimal window
print(">>> First pass: analyzing reward distribution...")
all_rewards = []
all_actions = []
all_dones = []
step_count = 0

start_time = time.time()
for o_t, a_t, r_tp1, o_tp1, done in ds:
    all_rewards.append(r_tp1.numpy())
    all_actions.append(a_t.numpy())
    all_dones.append(done.numpy())
    step_count += 1
    
    if step_count % 100000 == 0:
        elapsed = time.time() - start_time
        print(f"  Processed {step_count:,} steps in {elapsed:.1f}s")
    
    # Safety limit to prevent memory issues
    if step_count >= 2000000:  # 2M steps max
        print(f"  Reached safety limit of {step_count:,} steps")
        break

all_rewards = np.array(all_rewards)
all_actions = np.array(all_actions)
all_dones = np.array(all_dones)

print(f">>> Loaded {len(all_rewards):,} transitions")
print(f">>> Reward stats: min={all_rewards.min():.3f}, max={all_rewards.max():.3f}, mean={all_rewards.mean():.6f}")
print(f">>> Positive rewards: {np.sum(all_rewards > 0):,} ({np.sum(all_rewards > 0)/len(all_rewards)*100:.2f}%)")

# ---- Find optimal 150k window ----
print(f">>> Finding optimal {TARGET_LENGTH:,} consecutive frames...")

if len(all_rewards) < TARGET_LENGTH:
    print(f"  Dataset too small ({len(all_rewards):,} < {TARGET_LENGTH:,})")
    print(f"  Using all available data")
    best_start = 0
    best_positive_count = np.sum(all_rewards > 0)
else:
    # Use sliding window to find best 150k segment
    best_start = 0
    best_positive_count = 0
    
    print(f"  Scanning {len(all_rewards) - TARGET_LENGTH + 1:,} possible windows...")
    
    # Calculate positive rewards in first window
    current_positive_count = np.sum(all_rewards[:TARGET_LENGTH] > 0)
    best_positive_count = current_positive_count
    
    # Slide window and update count efficiently
    for start in range(1, len(all_rewards) - TARGET_LENGTH + 1):
        # Remove old element, add new element
        if all_rewards[start - 1] > 0:
            current_positive_count -= 1
        if all_rewards[start + TARGET_LENGTH - 1] > 0:
            current_positive_count += 1
        
        if current_positive_count > best_positive_count:
            best_positive_count = current_positive_count
            best_start = start
        
        if start % 50000 == 0:
            print(f"    Scanned {start:,} windows, best so far: {best_positive_count:,} positive rewards")

print(f">>> Best window found:")
print(f"  Start position: {best_start:,}")
print(f"  End position: {best_start + TARGET_LENGTH - 1:,}")
print(f"  Positive rewards: {best_positive_count:,} ({best_positive_count/TARGET_LENGTH*100:.2f}%)")
print(f"  Total reward: {np.sum(all_rewards[best_start:best_start + TARGET_LENGTH]):.1f}")

# ---- Load and save the optimal window ----
print(f">>> Loading optimal window data...")

# Reset dataset
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Skip to the optimal start position
print(f"  Skipping to position {best_start:,}...")
ds_skipped = ds.skip(best_start)

# Take exactly TARGET_LENGTH frames
ds_window = ds_skipped.take(TARGET_LENGTH)

# Collect the data
sampled_data = {
    'observations_t': [],
    'observations_tp1': [],
    'actions': [],
    'rewards': [],
    'dones': [],
    'steps': []
}

print(f"  Collecting {TARGET_LENGTH:,} frames...")
start_time = time.time()
for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(ds_window):
    sampled_data['observations_t'].append(o_t.numpy())
    sampled_data['observations_tp1'].append(o_tp1.numpy())
    sampled_data['actions'].append(a_t.numpy())
    sampled_data['rewards'].append(r_tp1.numpy())
    sampled_data['dones'].append(done.numpy())
    sampled_data['steps'].append(best_start + i)
    
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - start_time
        print(f"    Collected {i+1:,}/{TARGET_LENGTH:,} frames in {elapsed:.1f}s")

# Convert to numpy arrays
print(f">>> Converting to numpy arrays...")
for key in sampled_data:
    if key in ['observations_t', 'observations_tp1']:
        sampled_data[key] = np.array(sampled_data[key], dtype=np.uint8)
    else:
        sampled_data[key] = np.array(sampled_data[key])

# ---- Save the sampled data ----
print(f">>> Saving sampled data...")
os.makedirs("sampled_data", exist_ok=True)

# Save as pickle for easy loading
with open("sampled_data/optimal_150k_frames.pkl", "wb") as f:
    pickle.dump(sampled_data, f)

# Save metadata
metadata = {
    'total_frames': TARGET_LENGTH,
    'start_position': best_start,
    'end_position': best_start + TARGET_LENGTH - 1,
    'positive_rewards': int(best_positive_count),
    'positive_reward_percentage': float(best_positive_count / TARGET_LENGTH * 100),
    'total_reward': float(np.sum(sampled_data['rewards'])),
    'mean_reward': float(np.mean(sampled_data['rewards'])),
    'unique_actions': int(len(np.unique(sampled_data['actions']))),
    'episodes_ended': int(np.sum(sampled_data['dones'])),
    'observation_shape': sampled_data['observations_t'][0].shape,
    'dataset_info': {
        'shard_path': SHARD,
        'compression': COMPRESSION,
        'total_dataset_size': len(all_rewards)
    }
}

with open("sampled_data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Save as numpy arrays for direct access
np.savez_compressed(
    "sampled_data/optimal_150k_frames.npz",
    observations_t=sampled_data['observations_t'],
    observations_tp1=sampled_data['observations_tp1'],
    actions=sampled_data['actions'],
    rewards=sampled_data['rewards'],
    dones=sampled_data['dones'],
    steps=sampled_data['steps']
)

# ---- Print summary ----
print(f"\n" + "="*60)
print("SAMPLING COMPLETE!")
print("="*60)
print(f"📊 SAMPLED DATA SUMMARY:")
print(f"  Total frames: {TARGET_LENGTH:,}")
print(f"  Start position: {best_start:,}")
print(f"  End position: {best_start + TARGET_LENGTH - 1:,}")
print(f"  Positive rewards: {best_positive_count:,} ({best_positive_count/TARGET_LENGTH*100:.2f}%)")
print(f"  Total reward: {np.sum(sampled_data['rewards']):.1f}")
print(f"  Mean reward: {np.mean(sampled_data['rewards']):.6f}")
print(f"  Episodes ended: {np.sum(sampled_data['dones']):,}")
print(f"  Observation shape: {sampled_data['observations_t'][0].shape}")

print(f"\n💾 FILES CREATED:")
print(f"  - sampled_data/optimal_150k_frames.pkl: Full dataset (pickle)")
print(f"  - sampled_data/optimal_150k_frames.npz: Full dataset (numpy)")
print(f"  - sampled_data/metadata.pkl: Dataset metadata")

print(f"\n🎯 OPTIMIZATION RESULTS:")
print(f"  This sampling maximizes positive rewards in a consecutive 150k frame window")
print(f"  Found {best_positive_count:,} positive rewards out of {TARGET_LENGTH:,} total frames")
print(f"  This represents {best_positive_count/TARGET_LENGTH*100:.2f}% positive reward density")

print(f"\n✅ Ready for training! Load with:")
print(f"  import pickle")
print(f"  with open('sampled_data/optimal_150k_frames.pkl', 'rb') as f:")
print(f"      data = pickle.load(f)")
