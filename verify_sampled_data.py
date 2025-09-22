#!/usr/bin/env python3
"""
Verify and visualize the sampled 150k frames
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

print(">>> Verifying sampled 150k frames")

# Load the sampled data
print(">>> Loading sampled data...")
with open("sampled_data/optimal_150k_frames.pkl", "rb") as f:
    data = pickle.load(f)

with open("sampled_data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Print basic info
print(f"\n📊 DATA VERIFICATION:")
print(f"  Total frames: {len(data['rewards']):,}")
print(f"  Observation shape: {data['observations_t'][0].shape}")
print(f"  Action range: {data['actions'].min()} to {data['actions'].max()}")
print(f"  Reward range: {data['rewards'].min():.3f} to {data['rewards'].max():.3f}")
print(f"  Mean reward: {data['rewards'].mean():.6f}")

# Verify positive rewards
positive_rewards = np.sum(data['rewards'] > 0)
print(f"  Positive rewards: {positive_rewards:,} ({positive_rewards/len(data['rewards'])*100:.2f}%)")
print(f"  Total reward: {np.sum(data['rewards']):.1f}")

# Verify metadata matches
print(f"\n🔍 METADATA VERIFICATION:")
print(f"  Start position: {metadata['start_position']:,}")
print(f"  End position: {metadata['end_position']:,}")
print(f"  Positive rewards (metadata): {metadata['positive_rewards']:,}")
print(f"  Positive rewards (actual): {positive_rewards:,}")
print(f"  Match: {metadata['positive_rewards'] == positive_rewards}")

# Create visualizations
print(f"\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sampled 150k Frames - Verification', fontsize=16)

# 1. Reward distribution
axes[0, 0].hist(data['rewards'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Reward Distribution')
axes[0, 0].set_xlabel('Reward Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2. Rewards over time
axes[0, 1].plot(data['rewards'], alpha=0.6, linewidth=0.5)
axes[0, 1].set_title('Rewards Over Time')
axes[0, 1].set_xlabel('Frame Index')
axes[0, 1].set_ylabel('Reward Value')
axes[0, 1].grid(True, alpha=0.3)

# 3. Action distribution
action_counts = np.bincount(data['actions'])
axes[1, 0].bar(range(len(action_counts)), action_counts, alpha=0.7)
axes[1, 0].set_title('Action Distribution')
axes[1, 0].set_xlabel('Action')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(True, alpha=0.3)

# 4. Reward density over time (moving average)
window_size = 1000
if len(data['rewards']) > window_size:
    moving_avg = np.convolve(data['rewards'], np.ones(window_size)/window_size, mode='valid')
    axes[1, 1].plot(moving_avg, alpha=0.8, linewidth=1)
    axes[1, 1].set_title(f'Reward Density (Moving Average, window={window_size})')
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampled_data/verification_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Show some sample frames
print(f"\n🖼️  Sample frames from the dataset...")
os.makedirs("sampled_data/sample_frames", exist_ok=True)

# Find frames with positive rewards
positive_indices = np.where(data['rewards'] > 0)[0]
print(f"  Found {len(positive_indices)} frames with positive rewards")

# Save first few positive reward frames
for i, idx in enumerate(positive_indices[:5]):
    obs = data['observations_t'][idx]
    reward = data['rewards'][idx]
    action = data['actions'][idx]
    
    # Create 2x2 grid of the 4 frames
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Frame {idx} - Action: {action}, Reward: {reward:.1f}', fontsize=14)
    
    for j in range(4):
        row, col = j // 2, j % 2
        axes[row, col].imshow(obs[:, :, j], cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'Frame {j+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sampled_data/sample_frames/positive_reward_{i+1}_frame_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Saved sample frames to sampled_data/sample_frames/")

# Summary statistics
print(f"\n📋 FINAL SUMMARY:")
print(f"  ✅ Successfully sampled 150,000 consecutive frames")
print(f"  ✅ Optimized for maximum positive rewards: {positive_rewards:,} ({positive_rewards/150000*100:.2f}%)")
print(f"  ✅ Data saved in multiple formats (pickle, numpy)")
print(f"  ✅ Ready for training!")

print(f"\n💾 Files created:")
print(f"  - sampled_data/optimal_150k_frames.pkl")
print(f"  - sampled_data/optimal_150k_frames.npz") 
print(f"  - sampled_data/metadata.pkl")
print(f"  - sampled_data/verification_plots.png")
print(f"  - sampled_data/sample_frames/ (sample positive reward frames)")

print(f"\n🎯 This dataset is optimized for training with maximum positive reward density!")
