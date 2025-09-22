#!/usr/bin/env python3
"""
Visualize Latent World Models Dataset

This script provides comprehensive visualization capabilities for the latent world models
dataset files (.pth) containing Atari Breakout data. It can visualize sequences, statistics,
frame analysis, and create various plots to understand the dataset structure and content.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import argparse

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Note: seaborn not available, using matplotlib defaults")


# Define the ExperienceDataset class for loading
class ExperienceDataset:
    """Dataset class matching the latent-world-models repository requirements"""
    
    def __init__(self, states: list, actions: list, rewards: list, stop_episodes: list, 
                 sequence_length: int = 6):
        # Configuration (FIXED values as per repository requirements)
        self.sequence_length = sequence_length
        self.long_rolling_mode = False
        self.long_rolling_window = 5
        
        # Concatenate all data across episodes
        self.states = torch.cat(states, dim=0)  # [total_T, H, W] float32
        self.actions = torch.cat(actions, dim=0)  # [total_T] int32
        self.rewards = torch.cat(rewards, dim=0)  # [total_T] float32
        self.stop_episodes = torch.tensor(stop_episodes, dtype=torch.bool)  # [total_T] bool
        
        # Compute valid sequence start indices
        self.valid_start_indices = self._compute_valid_start_indices()
    
    def _compute_valid_start_indices(self) -> list:
        """Compute valid sequence start indices ensuring no episode boundaries crossed"""
        valid_indices = []
        N = len(self.stop_episodes)
        
        for start in range(N - self.sequence_length):
            # Check if sequence crosses episode boundary
            window = self.stop_episodes[start : start + self.sequence_length]
            if not window.any():  # No episode boundaries in window
                valid_indices.append(start)
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_start_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_start_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        return {
            'states': self.states[start_idx:end_idx],
            'actions': self.actions[start_idx:end_idx],
            'rewards': self.rewards[start_idx:end_idx],
            'stop_episodes': self.stop_episodes[start_idx:end_idx]
        }


class LatentWorldModelsVisualizer:
    """Visualizer for latent world models dataset files"""
    
    def __init__(self, data_dir: str = "latent_world_models_data"):
        """
        Initialize the visualizer
        
        Args:
            data_dir: Directory containing the .pth files
        """
        self.data_dir = data_dir
        self.datasets = {}
        self.metadata = {}
        
    def load_dataset(self, base_name: str = "atari_breakout_150k") -> Dict[str, any]:
        """
        Load all three dataset files (pretrain, world_model, reward_model)
        
        Args:
            base_name: Base name of the dataset files
            
        Returns:
            Dictionary containing loaded datasets and metadata
        """
        buckets = ['pretrain', 'world_model', 'reward_model']
        loaded_data = {}
        
        print(f"Loading dataset: {base_name}")
        print("=" * 50)
        
        for bucket in buckets:
            filename = os.path.join(self.data_dir, f"{base_name}_{bucket}.pth")
            
            if not os.path.exists(filename):
                print(f"❌ File not found: {filename}")
                continue
                
            print(f"📁 Loading {bucket} bucket...")
            try:
                # Load the data using torch.load
                data = torch.load(filename, map_location='cpu', weights_only=False)
                loaded_data[bucket] = data
                print(f"  ✅ Loaded {bucket}: {len(data['train_dataset'])} train, {len(data['val_dataset'])} val sequences")
            except Exception as e:
                print(f"  ❌ Error loading {bucket}: {e}")
                
        self.datasets = loaded_data
        return loaded_data
    
    def visualize_sequence(self, bucket: str, sequence_idx: int = 0, 
                          split: str = 'train', save_path: Optional[str] = None) -> None:
        """
        Visualize a single 6-timestep sequence
        
        Args:
            bucket: Which bucket to use ('pretrain', 'world_model', 'reward_model')
            sequence_idx: Index of sequence to visualize
            split: 'train' or 'val'
            save_path: Optional path to save the visualization
        """
        if bucket not in self.datasets:
            print(f"❌ Bucket {bucket} not loaded. Available: {list(self.datasets.keys())}")
            return
            
        dataset = self.datasets[bucket][f'{split}_dataset']
        
        if sequence_idx >= len(dataset):
            print(f"❌ Sequence index {sequence_idx} out of range. Max: {len(dataset)-1}")
            return
            
        # Get the sequence
        sequence = dataset[sequence_idx]
        states = sequence['states']  # [6, 84, 84]
        actions = sequence['actions']  # [6]
        rewards = sequence['rewards']  # [6]
        stop_episodes = sequence['stop_episodes']  # [6]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{bucket.upper()} - {split.upper()} Sequence {sequence_idx}', fontsize=16)
        
        # Plot each timestep
        for t in range(6):
            row, col = t // 3, t % 3
            ax = axes[row, col]
            
            # Display frame
            frame = states[t].numpy()
            ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
            
            # Add information
            action_names = ['NOOP', 'FIRE', 'UP', 'DOWN']
            action_name = action_names[actions[t].item()] if actions[t].item() < len(action_names) else f'Action {actions[t].item()}'
            
            title = f't={t}: {action_name}\nReward: {rewards[t].item():.1f}'
            if stop_episodes[t]:
                title += ' [EPISODE END]'
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Sequence visualization saved to {save_path}")
        
        plt.show()
        
        # Print sequence details
        print(f"\n📊 Sequence Details:")
        print(f"  Actions: {actions.tolist()}")
        print(f"  Rewards: {rewards.tolist()}")
        print(f"  Episode boundaries: {stop_episodes.tolist()}")
        print(f"  Total reward: {rewards.sum().item():.1f}")
        print(f"  Positive rewards: {(rewards > 0).sum().item()}")
    
    def visualize_frame_differences(self, bucket: str, sequence_idx: int = 0,
                                   split: str = 'train', save_path: Optional[str] = None) -> None:
        """
        Visualize frame differences to show motion/change between consecutive frames
        
        Args:
            bucket: Which bucket to use
            sequence_idx: Index of sequence to visualize
            split: 'train' or 'val'
            save_path: Optional path to save the visualization
        """
        if bucket not in self.datasets:
            print(f"❌ Bucket {bucket} not loaded")
            return
            
        dataset = self.datasets[bucket][f'{split}_dataset']
        sequence = dataset[sequence_idx]
        states = sequence['states'].numpy()  # [6, 84, 84]
        
        # Calculate differences between consecutive frames
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{bucket.upper()} - Frame Differences (Sequence {sequence_idx})', fontsize=16)
        
        # Plot original frames and differences
        for t in range(6):
            row, col = t // 3, t % 3
            ax = axes[row, col]
            
            if t == 0:
                # First frame - show original
                ax.imshow(states[t], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f't={t} (Original)', fontsize=10)
            else:
                # Show difference from previous frame
                diff = states[t] - states[t-1]
                im = ax.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
                ax.set_title(f't={t} (Diff from t-1)', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Frame differences saved to {save_path}")
        
        plt.show()
    
    def plot_reward_statistics(self, bucket: str, save_path: Optional[str] = None) -> None:
        """
        Plot reward statistics for a bucket
        
        Args:
            bucket: Which bucket to analyze
            save_path: Optional path to save the plot
        """
        if bucket not in self.datasets:
            print(f"❌ Bucket {bucket} not loaded")
            return
            
        train_dataset = self.datasets[bucket]['train_dataset']
        val_dataset = self.datasets[bucket]['val_dataset']
        
        train_rewards = train_dataset.rewards.numpy()
        val_rewards = val_dataset.rewards.numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{bucket.upper()} - Reward Statistics', fontsize=16)
        
        # Reward distribution
        ax1 = axes[0, 0]
        ax1.hist(train_rewards, bins=50, alpha=0.7, label='Train', color='blue')
        ax1.hist(val_rewards, bins=50, alpha=0.7, label='Val', color='orange')
        ax1.set_xlabel('Reward Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positive vs negative rewards
        ax2 = axes[0, 1]
        train_positive = (train_rewards > 0).sum()
        train_negative = (train_rewards <= 0).sum()
        val_positive = (val_rewards > 0).sum()
        val_negative = (val_rewards <= 0).sum()
        
        categories = ['Positive', 'Negative']
        train_counts = [train_positive, train_negative]
        val_counts = [val_positive, val_negative]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, train_counts, width, label='Train', color='blue', alpha=0.7)
        ax2.bar(x + width/2, val_counts, width, label='Val', color='orange', alpha=0.7)
        ax2.set_xlabel('Reward Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Positive vs Negative Rewards')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reward over time (first 1000 transitions)
        ax3 = axes[1, 0]
        ax3.plot(train_rewards[:1000], alpha=0.7, label='Train', color='blue')
        ax3.plot(val_rewards[:1000], alpha=0.7, label='Val', color='orange')
        ax3.set_xlabel('Transition Index')
        ax3.set_ylabel('Reward')
        ax3.set_title('Rewards Over Time (First 1000)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Train Statistics:
        • Total transitions: {len(train_rewards):,}
        • Positive rewards: {train_positive:,} ({train_positive/len(train_rewards)*100:.2f}%)
        • Mean reward: {train_rewards.mean():.3f}
        • Std reward: {train_rewards.std():.3f}
        • Max reward: {train_rewards.max():.1f}
        • Min reward: {train_rewards.min():.1f}
        
        Val Statistics:
        • Total transitions: {len(val_rewards):,}
        • Positive rewards: {val_positive:,} ({val_positive/len(val_rewards)*100:.2f}%)
        • Mean reward: {val_rewards.mean():.3f}
        • Std reward: {val_rewards.std():.3f}
        • Max reward: {val_rewards.max():.1f}
        • Min reward: {val_rewards.min():.1f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Reward statistics saved to {save_path}")
        
        plt.show()
    
    def plot_action_statistics(self, bucket: str, save_path: Optional[str] = None) -> None:
        """
        Plot action statistics for a bucket
        
        Args:
            bucket: Which bucket to analyze
            save_path: Optional path to save the plot
        """
        if bucket not in self.datasets:
            print(f"❌ Bucket {bucket} not loaded")
            return
            
        train_dataset = self.datasets[bucket]['train_dataset']
        val_dataset = self.datasets[bucket]['val_dataset']
        
        train_actions = train_dataset.actions.numpy()
        val_actions = val_dataset.actions.numpy()
        
        # Filter out padding actions (-1)
        train_actions_valid = train_actions[train_actions != -1]
        val_actions_valid = val_actions[val_actions != -1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{bucket.upper()} - Action Statistics', fontsize=16)
        
        # Action distribution
        ax1 = axes[0, 0]
        action_names = ['NOOP', 'FIRE', 'UP', 'DOWN']
        train_counts = [np.sum(train_actions_valid == i) for i in range(4)]
        val_counts = [np.sum(val_actions_valid == i) for i in range(4)]
        
        x = np.arange(len(action_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_counts, width, label='Train', color='blue', alpha=0.7)
        ax1.bar(x + width/2, val_counts, width, label='Val', color='orange', alpha=0.7)
        ax1.set_xlabel('Action')
        ax1.set_ylabel('Count')
        ax1.set_title('Action Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(action_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Action over time (first 1000 transitions)
        ax2 = axes[0, 1]
        ax2.plot(train_actions[:1000], alpha=0.7, label='Train', color='blue', marker='o', markersize=2)
        ax2.plot(val_actions[:1000], alpha=0.7, label='Val', color='orange', marker='s', markersize=2)
        ax2.set_xlabel('Transition Index')
        ax2.set_ylabel('Action')
        ax2.set_title('Actions Over Time (First 1000)')
        ax2.set_yticks(range(4))
        ax2.set_yticklabels(action_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Action transition matrix
        ax3 = axes[1, 0]
        # Create transition matrix for train data
        transitions = np.zeros((4, 4))
        for i in range(len(train_actions_valid) - 1):
            if train_actions_valid[i] != -1 and train_actions_valid[i+1] != -1:
                transitions[train_actions_valid[i], train_actions_valid[i+1]] += 1
        
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions_norm = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
        
        im = ax3.imshow(transitions_norm, cmap='Blues')
        ax3.set_xlabel('Next Action')
        ax3.set_ylabel('Current Action')
        ax3.set_title('Action Transition Probabilities (Train)')
        ax3.set_xticks(range(4))
        ax3.set_yticks(range(4))
        ax3.set_xticklabels(action_names)
        ax3.set_yticklabels(action_names)
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax3.text(j, i, f'{transitions_norm[i, j]:.2f}',
                               ha="center", va="center", color="black" if transitions_norm[i, j] < 0.5 else "white")
        
        plt.colorbar(im, ax=ax3)
        
        # Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Train Statistics:
        • Total actions: {len(train_actions):,}
        • Valid actions: {len(train_actions_valid):,}
        • Padding actions: {np.sum(train_actions == -1):,}
        
        Action Counts:
        • NOOP: {train_counts[0]:,} ({train_counts[0]/len(train_actions_valid)*100:.1f}%)
        • FIRE: {train_counts[1]:,} ({train_counts[1]/len(train_actions_valid)*100:.1f}%)
        • UP: {train_counts[2]:,} ({train_counts[2]/len(train_actions_valid)*100:.1f}%)
        • DOWN: {train_counts[3]:,} ({train_counts[3]/len(train_actions_valid)*100:.1f}%)
        
        Val Statistics:
        • Total actions: {len(val_actions):,}
        • Valid actions: {len(val_actions_valid):,}
        • Padding actions: {np.sum(val_actions == -1):,}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Action statistics saved to {save_path}")
        
        plt.show()
    
    def create_dataset_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive summary of all loaded datasets
        
        Args:
            save_path: Optional path to save the summary
        """
        if not self.datasets:
            print("❌ No datasets loaded")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Latent World Models Dataset Summary', fontsize=16)
        
        buckets = list(self.datasets.keys())
        colors = ['blue', 'orange', 'green']
        
        # Dataset sizes
        ax1 = axes[0, 0]
        train_sizes = [len(self.datasets[bucket]['train_dataset']) for bucket in buckets]
        val_sizes = [len(self.datasets[bucket]['val_dataset']) for bucket in buckets]
        
        x = np.arange(len(buckets))
        width = 0.35
        
        ax1.bar(x - width/2, train_sizes, width, label='Train', color='blue', alpha=0.7)
        ax1.bar(x + width/2, val_sizes, width, label='Val', color='orange', alpha=0.7)
        ax1.set_xlabel('Bucket')
        ax1.set_ylabel('Number of Sequences')
        ax1.set_title('Dataset Sizes')
        ax1.set_xticks(x)
        ax1.set_xticklabels([b.upper() for b in buckets])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total transitions
        ax2 = axes[0, 1]
        train_transitions = [len(self.datasets[bucket]['train_dataset'].states) for bucket in buckets]
        val_transitions = [len(self.datasets[bucket]['val_dataset'].states) for bucket in buckets]
        
        ax2.bar(x - width/2, train_transitions, width, label='Train', color='blue', alpha=0.7)
        ax2.bar(x + width/2, val_transitions, width, label='Val', color='orange', alpha=0.7)
        ax2.set_xlabel('Bucket')
        ax2.set_ylabel('Number of Transitions')
        ax2.set_title('Total Transitions')
        ax2.set_xticks(x)
        ax2.set_xticklabels([b.upper() for b in buckets])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Positive reward percentages
        ax3 = axes[0, 2]
        train_positive_pct = []
        val_positive_pct = []
        
        for bucket in buckets:
            train_rewards = self.datasets[bucket]['train_dataset'].rewards
            val_rewards = self.datasets[bucket]['val_dataset'].rewards
            train_positive_pct.append((train_rewards > 0).sum().item() / len(train_rewards) * 100)
            val_positive_pct.append((val_rewards > 0).sum().item() / len(val_rewards) * 100)
        
        ax3.bar(x - width/2, train_positive_pct, width, label='Train', color='blue', alpha=0.7)
        ax3.bar(x + width/2, val_positive_pct, width, label='Val', color='orange', alpha=0.7)
        ax3.set_xlabel('Bucket')
        ax3.set_ylabel('Positive Reward %')
        ax3.set_title('Positive Reward Percentage')
        ax3.set_xticks(x)
        ax3.set_xticklabels([b.upper() for b in buckets])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Reward distributions (combined)
        ax4 = axes[1, 0]
        for i, bucket in enumerate(buckets):
            train_rewards = self.datasets[bucket]['train_dataset'].rewards.numpy()
            ax4.hist(train_rewards, bins=30, alpha=0.5, label=f'{bucket.upper()} Train', color=colors[i])
        ax4.set_xlabel('Reward Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reward Distributions (Train)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Action distributions (combined)
        ax5 = axes[1, 1]
        action_names = ['NOOP', 'FIRE', 'UP', 'DOWN']
        for i, bucket in enumerate(buckets):
            actions = self.datasets[bucket]['train_dataset'].actions.numpy()
            valid_actions = actions[actions != -1]
            action_counts = [np.sum(valid_actions == j) for j in range(4)]
            ax5.plot(action_names, action_counts, marker='o', label=f'{bucket.upper()}', color=colors[i])
        ax5.set_xlabel('Action')
        ax5.set_ylabel('Count')
        ax5.set_title('Action Distributions (Train)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Metadata summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "Dataset Summary:\n\n"
        for bucket in buckets:
            metadata = self.datasets[bucket]['metadata']
            summary_text += f"{bucket.upper()}:\n"
            summary_text += f"• Environment: {metadata['environment_name']}\n"
            summary_text += f"• Collection: {metadata['collection_method']}\n"
            summary_text += f"• Image size: {metadata['image_height']}x{metadata['image_width']}\n"
            summary_text += f"• Val split: {metadata['validation_split_ratio']:.1%}\n\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Dataset summary saved to {save_path}")
        
        plt.show()
    
    def save_sample_frames(self, bucket: str, num_samples: int = 5, 
                          split: str = 'train', output_dir: str = 'sample_frames') -> None:
        """
        Save sample frames from sequences as individual images
        
        Args:
            bucket: Which bucket to sample from
            num_samples: Number of sequences to sample
            split: 'train' or 'val'
            output_dir: Directory to save images
        """
        if bucket not in self.datasets:
            print(f"❌ Bucket {bucket} not loaded")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        dataset = self.datasets[bucket][f'{split}_dataset']
        
        print(f"💾 Saving {num_samples} sample sequences from {bucket} {split}...")
        
        for i in range(min(num_samples, len(dataset))):
            sequence = dataset[i]
            states = sequence['states'].numpy()  # [6, 84, 84]
            actions = sequence['actions'].numpy()
            rewards = sequence['rewards'].numpy()
            
            # Create a 2x3 grid of frames
            combined = np.zeros((84*2, 84*3), dtype=np.uint8)
            for t in range(6):
                row, col = t // 3, t % 3
                y_start, y_end = row * 84, (row + 1) * 84
                x_start, x_end = col * 84, (col + 1) * 84
                combined[y_start:y_end, x_start:x_end] = (states[t] * 255).astype(np.uint8)
            
            # Save as image
            img = Image.fromarray(combined, mode='L')
            filename = f"{output_dir}/{bucket}_{split}_sequence_{i:03d}.png"
            img.save(filename)
            
            # Save metadata
            metadata_filename = f"{output_dir}/{bucket}_{split}_sequence_{i:03d}_info.txt"
            with open(metadata_filename, 'w') as f:
                f.write(f"Sequence {i} from {bucket} {split}\n")
                f.write(f"Actions: {actions.tolist()}\n")
                f.write(f"Rewards: {rewards.tolist()}\n")
                f.write(f"Total reward: {rewards.sum():.1f}\n")
                f.write(f"Positive rewards: {(rewards > 0).sum()}\n")
        
        print(f"✅ Saved {num_samples} sample sequences to {output_dir}/")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Visualize Latent World Models Dataset')
    parser.add_argument('--data_dir', default='latent_world_models_data', 
                       help='Directory containing .pth files')
    parser.add_argument('--base_name', default='atari_breakout_150k',
                       help='Base name of dataset files')
    parser.add_argument('--bucket', choices=['pretrain', 'world_model', 'reward_model'],
                       help='Specific bucket to visualize')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Sequence index to visualize')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                       help='Dataset split to use')
    parser.add_argument('--output_dir', default='visualization_output',
                       help='Output directory for saved images')
    parser.add_argument('--save_samples', action='store_true',
                       help='Save sample frames as images')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample sequences to save')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = LatentWorldModelsVisualizer(args.data_dir)
    
    # Load datasets
    visualizer.load_dataset(args.base_name)
    
    if not visualizer.datasets:
        print("❌ No datasets loaded successfully")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize based on arguments
    if args.bucket:
        # Visualize specific bucket
        print(f"\n🎨 Visualizing {args.bucket} bucket...")
        
        # Sequence visualization
        visualizer.visualize_sequence(
            args.bucket, args.sequence_idx, args.split,
            save_path=os.path.join(args.output_dir, f'{args.bucket}_sequence_{args.sequence_idx}.png')
        )
        
        # Frame differences
        visualizer.visualize_frame_differences(
            args.bucket, args.sequence_idx, args.split,
            save_path=os.path.join(args.output_dir, f'{args.bucket}_differences_{args.sequence_idx}.png')
        )
        
        # Reward statistics
        visualizer.plot_reward_statistics(
            args.bucket,
            save_path=os.path.join(args.output_dir, f'{args.bucket}_rewards.png')
        )
        
        # Action statistics
        visualizer.plot_action_statistics(
            args.bucket,
            save_path=os.path.join(args.output_dir, f'{args.bucket}_actions.png')
        )
        
        # Save sample frames
        if args.save_samples:
            visualizer.save_sample_frames(
                args.bucket, args.num_samples, args.split,
                os.path.join(args.output_dir, f'{args.bucket}_samples')
            )
    else:
        # Create comprehensive summary
        print("\n📊 Creating comprehensive dataset summary...")
        visualizer.create_dataset_summary(
            save_path=os.path.join(args.output_dir, 'dataset_summary.png')
        )
        
        # Save sample frames from all buckets
        if args.save_samples:
            for bucket in visualizer.datasets.keys():
                visualizer.save_sample_frames(
                    bucket, args.num_samples, args.split,
                    os.path.join(args.output_dir, f'{bucket}_samples')
                )
    
    print(f"\n✅ Visualization complete! Check {args.output_dir}/ for saved images.")


if __name__ == "__main__":
    main()
