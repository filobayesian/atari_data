#!/usr/bin/env python3
"""
Validate the adapted latent-world-models dataset files
"""
import os
import torch
import numpy as np

print(">>> Validating latent-world-models dataset files")

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

def validate_dataset_files(base_name: str = "atari_breakout_150k") -> None:
    """
    Validate the adapted dataset files
    """
    print(f">>> Validating dataset files for {base_name}")
    
    required_files = [
        f"latent_world_models_data/{base_name}_pretrain.pth",
        f"latent_world_models_data/{base_name}_world_model.pth",
        f"latent_world_models_data/{base_name}_reward_model.pth"
    ]
    
    all_valid = True
    
    for filename in required_files:
        print(f"\n--- Validating {filename} ---")
        
        if not os.path.exists(filename):
            print(f"  ❌ Missing file: {filename}")
            all_valid = False
            continue
        
        print(f"  ✅ Found: {filename}")
        
        # Load and validate structure
        try:
            payload = torch.load(filename, map_location='cpu', weights_only=False)
            
            # Check required keys
            required_keys = ['train_dataset', 'val_dataset', 'metadata']
            print(f"  📋 Checking required keys...")
            for key in required_keys:
                if key not in payload:
                    print(f"    ❌ Missing key: {key}")
                    all_valid = False
                else:
                    print(f"    ✅ Has key: {key}")
            
            # Validate metadata
            if 'metadata' in payload:
                metadata = payload['metadata']
                print(f"  📊 Validating metadata...")
                required_metadata = [
                    'environment_name', 'collection_method', 'bucket',
                    'image_height', 'image_width', 'validation_split_ratio',
                    'num_train_transitions', 'num_val_transitions'
                ]
                
                for key in required_metadata:
                    if key not in metadata:
                        print(f"    ❌ Missing metadata: {key}")
                        all_valid = False
                    else:
                        print(f"    ✅ {key}: {metadata[key]}")
            
            # Validate dataset structure
            if 'train_dataset' in payload:
                train_dataset = payload['train_dataset']
                print(f"  🔍 Validating train dataset structure...")
                print(f"    States shape: {train_dataset.states.shape}")
                print(f"    Actions shape: {train_dataset.actions.shape}")
                print(f"    Rewards shape: {train_dataset.rewards.shape}")
                print(f"    Stop episodes shape: {train_dataset.stop_episodes.shape}")
                print(f"    Valid sequences: {len(train_dataset.valid_start_indices):,}")
                print(f"    Sequence length: {train_dataset.sequence_length}")
                
                # Validate data types and ranges
                print(f"  🔬 Validating data types and ranges...")
                
                # States validation
                if train_dataset.states.dtype != torch.float32:
                    print(f"    ❌ States dtype should be float32, got {train_dataset.states.dtype}")
                    all_valid = False
                else:
                    print(f"    ✅ States dtype: {train_dataset.states.dtype}")
                
                if train_dataset.states.min() < 0.0 or train_dataset.states.max() > 1.0:
                    print(f"    ❌ States range should be [0,1], got [{train_dataset.states.min():.3f}, {train_dataset.states.max():.3f}]")
                    all_valid = False
                else:
                    print(f"    ✅ States range: [{train_dataset.states.min():.3f}, {train_dataset.states.max():.3f}]")
                
                # Actions validation
                if train_dataset.actions.dtype != torch.int32:
                    print(f"    ❌ Actions dtype should be int32, got {train_dataset.actions.dtype}")
                    all_valid = False
                else:
                    print(f"    ✅ Actions dtype: {train_dataset.actions.dtype}")
                
                # Check for padding values
                pad_actions = (train_dataset.actions == -1).sum().item()
                pad_rewards = (train_dataset.rewards == -100.0).sum().item()
                print(f"    ✅ Padding actions (-1): {pad_actions}")
                print(f"    ✅ Padding rewards (-100.0): {pad_rewards}")
                
                # Rewards validation
                if train_dataset.rewards.dtype != torch.float32:
                    print(f"    ❌ Rewards dtype should be float32, got {train_dataset.rewards.dtype}")
                    all_valid = False
                else:
                    print(f"    ✅ Rewards dtype: {train_dataset.rewards.dtype}")
                
                # Stop episodes validation
                if train_dataset.stop_episodes.dtype != torch.bool:
                    print(f"    ❌ Stop episodes dtype should be bool, got {train_dataset.stop_episodes.dtype}")
                    all_valid = False
                else:
                    print(f"    ✅ Stop episodes dtype: {train_dataset.stop_episodes.dtype}")
                
                # Episode boundaries validation
                episode_boundaries = train_dataset.stop_episodes.sum().item()
                print(f"    ✅ Episode boundaries: {episode_boundaries}")
                
                # Test sequence sampling
                print(f"  🧪 Testing sequence sampling...")
                try:
                    sample = train_dataset[0]
                    print(f"    ✅ Sample sequence shape: states={sample['states'].shape}, actions={sample['actions'].shape}")
                    print(f"    ✅ Sample rewards: {sample['rewards'].tolist()}")
                    print(f"    ✅ Sample actions: {sample['actions'].tolist()}")
                except Exception as e:
                    print(f"    ❌ Error sampling sequence: {e}")
                    all_valid = False
            
            if 'val_dataset' in payload:
                val_dataset = payload['val_dataset']
                print(f"  🔍 Validating val dataset structure...")
                print(f"    States shape: {val_dataset.states.shape}")
                print(f"    Valid sequences: {len(val_dataset.valid_start_indices):,}")
            
            print(f"  ✅ {filename} validation completed successfully")
            
        except Exception as e:
            print(f"    ❌ Error loading {filename}: {e}")
            all_valid = False
    
    print(f"\n{'='*60}")
    if all_valid:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Dataset is ready for use with latent-world-models repository")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please check the errors above and fix them")
    print(f"{'='*60}")

def print_dataset_summary(base_name: str = "atari_breakout_150k") -> None:
    """Print a summary of the dataset"""
    print(f"\n📊 DATASET SUMMARY: {base_name}")
    print(f"{'='*50}")
    
    buckets = ['pretrain', 'world_model', 'reward_model']
    
    for bucket in buckets:
        filename = f"latent_world_models_data/{base_name}_{bucket}.pth"
        
        if os.path.exists(filename):
            try:
                payload = torch.load(filename, map_location='cpu', weights_only=False)
                train_dataset = payload['train_dataset']
                val_dataset = payload['val_dataset']
                metadata = payload['metadata']
                
                print(f"\n📁 {bucket.upper()} BUCKET:")
                print(f"  Train sequences: {len(train_dataset):,}")
                print(f"  Val sequences: {len(val_dataset):,}")
                print(f"  Train transitions: {len(train_dataset.states):,}")
                print(f"  Val transitions: {len(val_dataset.states):,}")
                print(f"  Environment: {metadata['environment_name']}")
                print(f"  Collection method: {metadata['collection_method']}")
                print(f"  Image size: {metadata['image_height']}x{metadata['image_width']}")
                
                # Calculate positive rewards
                train_positive = (train_dataset.rewards > 0).sum().item()
                val_positive = (val_dataset.rewards > 0).sum().item()
                print(f"  Train positive rewards: {train_positive:,} ({train_positive/len(train_dataset.states)*100:.2f}%)")
                print(f"  Val positive rewards: {val_positive:,} ({val_positive/len(val_dataset.states)*100:.2f}%)")
                
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")

if __name__ == "__main__":
    # Validate all files
    validate_dataset_files("atari_breakout_150k")
    
    # Print summary
    print_dataset_summary("atari_breakout_150k")
