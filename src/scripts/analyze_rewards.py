#!/usr/bin/env python3
"""
Comprehensive reward distribution analysis for Atari dataset
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

print(">>> Analyzing reward distribution in Atari dataset")

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

# ---- Load dataset ----
print(">>> Loading dataset...")
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Convert to list for analysis (this might take a while for large datasets)
print(">>> Converting to numpy arrays...")
all_data = []
for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(ds):
    all_data.append({
        'action': a_t.numpy(),
        'reward': r_tp1.numpy(),
        'done': done.numpy(),
        'step': i
    })
    if i % 10000 == 0:
        print(f"  Processed {i} transitions...")

print(f">>> Loaded {len(all_data)} transitions")

# ---- Extract data ----
rewards = np.array([d['reward'] for d in all_data])
actions = np.array([d['action'] for d in all_data])
dones = np.array([d['done'] for d in all_data])
steps = np.array([d['step'] for d in all_data])

# ---- Basic Statistics ----
print("\n" + "="*60)
print("REWARD DISTRIBUTION ANALYSIS")
print("="*60)

print(f"\n📊 BASIC STATISTICS:")
print(f"  Total transitions: {len(rewards):,}")
print(f"  Unique reward values: {len(np.unique(rewards))}")
print(f"  Reward range: {rewards.min():.3f} to {rewards.max():.3f}")
print(f"  Mean reward: {rewards.mean():.6f}")
print(f"  Median reward: {np.median(rewards):.6f}")
print(f"  Std deviation: {rewards.std():.6f}")

# ---- Reward Value Analysis ----
print(f"\n🎯 REWARD VALUES:")
unique_rewards, counts = np.unique(rewards, return_counts=True)
for reward, count in zip(unique_rewards, counts):
    percentage = (count / len(rewards)) * 100
    print(f"  {reward:6.1f}: {count:8,} occurrences ({percentage:5.2f}%)")

# ---- Episode Analysis ----
print(f"\n🎮 EPISODE ANALYSIS:")
episode_starts = np.where(np.concatenate([[True], dones[:-1]]))[0]
episode_ends = np.where(dones)[0]

print(f"  Total episodes: {len(episode_starts)}")
print(f"  Episodes that ended: {len(episode_ends)}")

if len(episode_ends) > 0:
    episode_lengths = episode_ends - episode_starts + 1
    print(f"  Episode length range: {episode_lengths.min()} to {episode_lengths.max()}")
    print(f"  Mean episode length: {episode_lengths.mean():.1f}")
    print(f"  Median episode length: {np.median(episode_lengths):.1f}")
    
    # Calculate episode rewards
    episode_rewards = []
    for start, end in zip(episode_starts, episode_ends):
        episode_reward = rewards[start:end+1].sum()
        episode_rewards.append(episode_reward)
    
    episode_rewards = np.array(episode_rewards)
    print(f"\n  Episode reward range: {episode_rewards.min():.1f} to {episode_rewards.max():.1f}")
    print(f"  Mean episode reward: {episode_rewards.mean():.2f}")
    print(f"  Median episode reward: {np.median(episode_rewards):.2f}")
else:
    print(f"  No episodes ended in this dataset (all dones=False)")
    print(f"  Treating entire dataset as one continuous sequence")
    episode_lengths = np.array([len(rewards)])
    episode_rewards = np.array([rewards.sum()])
    print(f"  Sequence length: {len(rewards):,} transitions")
    print(f"  Total reward: {rewards.sum():.1f}")

# ---- Action-Reward Analysis ----
print(f"\n🎮 ACTION-REWARD ANALYSIS:")
action_rewards = {}
for action in np.unique(actions):
    action_mask = actions == action
    action_reward_values = rewards[action_mask]
    action_rewards[action] = action_reward_values
    
    print(f"  Action {action}:")
    print(f"    Count: {len(action_reward_values):,} ({len(action_reward_values)/len(rewards)*100:.1f}%)")
    print(f"    Mean reward: {action_reward_values.mean():.6f}")
    print(f"    Unique rewards: {np.unique(action_reward_values)}")

# ---- Reward Distribution Visualization ----
print(f"\n📈 CREATING VISUALIZATIONS...")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Atari Breakout - Reward Distribution Analysis', fontsize=16)

# 1. Reward histogram
axes[0, 0].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Reward Distribution (All Transitions)')
axes[0, 0].set_xlabel('Reward Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2. Reward over time
axes[0, 1].plot(steps, rewards, alpha=0.6, linewidth=0.5)
axes[0, 1].set_title('Rewards Over Time')
axes[0, 1].set_xlabel('Transition Step')
axes[0, 1].set_ylabel('Reward Value')
axes[0, 1].grid(True, alpha=0.3)

# 3. Episode rewards
axes[0, 2].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black', color='green')
axes[0, 2].set_title('Episode Total Rewards')
axes[0, 2].set_xlabel('Episode Reward')
axes[0, 2].set_ylabel('Number of Episodes')
axes[0, 2].grid(True, alpha=0.3)

# 4. Action-reward box plot
action_data = [action_rewards[action] for action in sorted(action_rewards.keys())]
action_labels = [f'Action {action}' for action in sorted(action_rewards.keys())]
axes[1, 0].boxplot(action_data, labels=action_labels)
axes[1, 0].set_title('Reward Distribution by Action')
axes[1, 0].set_ylabel('Reward Value')
axes[1, 0].grid(True, alpha=0.3)

# 5. Episode length distribution
axes[1, 1].hist(episode_lengths, bins=30, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].set_title('Episode Length Distribution')
axes[1, 1].set_xlabel('Episode Length (steps)')
axes[1, 1].set_ylabel('Number of Episodes')
axes[1, 1].grid(True, alpha=0.3)

# 6. Reward value counts (bar chart)
axes[1, 2].bar(unique_rewards, counts, alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Reward Value Counts')
axes[1, 2].set_xlabel('Reward Value')
axes[1, 2].set_ylabel('Count')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reward_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- Advanced Analysis ----
print(f"\n🔍 ADVANCED ANALYSIS:")

# Reward sparsity
zero_rewards = np.sum(rewards == 0)
print(f"  Zero rewards: {zero_rewards:,} ({zero_rewards/len(rewards)*100:.2f}%)")
print(f"  Non-zero rewards: {len(rewards) - zero_rewards:,} ({(len(rewards) - zero_rewards)/len(rewards)*100:.2f}%)")

# Consecutive zero rewards
consecutive_zeros = []
current_zeros = 0
for reward in rewards:
    if reward == 0:
        current_zeros += 1
    else:
        if current_zeros > 0:
            consecutive_zeros.append(current_zeros)
        current_zeros = 0
if current_zeros > 0:
    consecutive_zeros.append(current_zeros)

if consecutive_zeros:
    print(f"  Max consecutive zeros: {max(consecutive_zeros)}")
    print(f"  Mean consecutive zeros: {np.mean(consecutive_zeros):.1f}")

# Reward patterns
print(f"\n  Reward patterns:")
for reward in unique_rewards:
    if reward != 0:
        # Find where this reward occurs
        reward_positions = np.where(rewards == reward)[0]
        if len(reward_positions) > 1:
            intervals = np.diff(reward_positions)
            print(f"    Reward {reward}: avg interval {intervals.mean():.1f} steps")

# ---- Save detailed data ----
print(f"\n💾 SAVING ANALYSIS DATA...")

# Create summary DataFrame
summary_data = {
    'metric': [
        'Total transitions', 'Unique rewards', 'Min reward', 'Max reward',
        'Mean reward', 'Median reward', 'Std reward', 'Zero rewards',
        'Non-zero rewards', 'Total episodes', 'Mean episode length',
        'Mean episode reward', 'Max episode reward'
    ],
    'value': [
        len(rewards), len(unique_rewards), rewards.min(), rewards.max(),
        rewards.mean(), np.median(rewards), rewards.std(), zero_rewards,
        len(rewards) - zero_rewards, len(episode_starts), episode_lengths.mean(),
        episode_rewards.mean(), episode_rewards.max()
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('reward_analysis_summary.csv', index=False)

# Save detailed transition data
df_transitions = pd.DataFrame({
    'step': steps,
    'action': actions,
    'reward': rewards,
    'done': dones
})
df_transitions.to_csv('reward_analysis_transitions.csv', index=False)

print(f"  ✅ Saved reward_analysis.png")
print(f"  ✅ Saved reward_analysis_summary.csv")
print(f"  ✅ Saved reward_analysis_transitions.csv")

print(f"\n🎉 REWARD ANALYSIS COMPLETE!")
print(f"   This analysis shows the reward structure of your Breakout dataset.")
print(f"   Understanding reward patterns is crucial for RL training!")
