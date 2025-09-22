"""
Convert RLU Atari Data to Latent World Models Format

This script converts data from the RLU Atari dataset (PNG format) to the format
expected by the latent-world-models repository.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import io
from typing import Tuple, List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ExperienceDataset


def decode_png_observation(png_obs: np.ndarray) -> np.ndarray:
    """
    Decode PNG-encoded observation to pixel array.
    
    Args:
        png_obs: Array of PNG byte strings, shape (4,)
        
    Returns:
        Decoded observation, shape (84, 84, 4)
    """
    decoded_frames = []
    for png_bytes in png_obs:
        # Decode PNG bytes to image
        img = Image.open(io.BytesIO(png_bytes))
        # Convert to numpy array and normalize to [0, 1]
        frame = np.array(img, dtype=np.uint8)
        decoded_frames.append(frame)
    
    # Stack frames along channel dimension: (4, 84, 84) -> (84, 84, 4)
    return np.stack(decoded_frames, axis=-1)


def convert_rlu_data_to_lwm_format(
    rlu_data_path: str,
    output_dir: str = "datasets",
    base_filename: str = "atari_breakout_150k",
    sequence_length: int = 6,
    validation_split: float = 0.2
) -> None:
    """
    Convert RLU Atari data to the format expected by latent-world-models.
    """
    
    print("Loading RLU data...")
    
    # Load RLU data
    if rlu_data_path.endswith('.pkl'):
        with open(rlu_data_path, 'rb') as f:
            data = pickle.load(f)
    elif rlu_data_path.endswith('.npz'):
        data = np.load(rlu_data_path, allow_pickle=True)
    else:
        raise ValueError("Data file must be .pkl or .npz format")
    
    print(f"Loaded data with {len(data['observations_t_png'])} frames")
    
    # Decode PNG observations
    print("Decoding PNG observations...")
    states = []
    actions = []
    rewards = []
    stop_episodes = []
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    total_frames = len(data['observations_t_png'])
    
    for i in range(0, total_frames, batch_size):
        end_idx = min(i + batch_size, total_frames)
        batch_obs_t = data['observations_t_png'][i:end_idx]
        batch_actions = data['actions'][i:end_idx]
        batch_rewards = data['rewards'][i:end_idx]
        batch_dones = data['dones'][i:end_idx]
        
        for obs_t, action, reward, done in zip(batch_obs_t, batch_actions, batch_rewards, batch_dones):
            # Decode PNG observation
            decoded_t = decode_png_observation(obs_t)  # (84, 84, 4)
            
            # Convert to torch tensor with correct shape for ExperienceDataset
            # ExperienceDataset expects states as [C, T_i, H, W] where C=1, T_i=4
            state = torch.from_numpy(decoded_t).permute(2, 0, 1).float()  # [4, 84, 84]
            state = state.unsqueeze(0)  # [1, 4, 84, 84] - add channel dimension
            
            states.append(state)
            actions.append(torch.tensor(action, dtype=torch.int32))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            stop_episodes.append(bool(done))
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i}/{total_frames} frames...")
    
    print(f"Decoded {len(states)} states")
    
    # Split data into three buckets
    total_frames = len(states)
    pretrain_end = total_frames // 3
    world_model_end = 2 * total_frames // 3
    
    pretrain_data = {
        'states': states[:pretrain_end],
        'actions': actions[:pretrain_end],
        'rewards': rewards[:pretrain_end],
        'stop_episodes': stop_episodes[:pretrain_end]
    }
    
    world_model_data = {
        'states': states[pretrain_end:world_model_end],
        'actions': actions[pretrain_end:world_model_end],
        'rewards': rewards[pretrain_end:world_model_end],
        'stop_episodes': stop_episodes[pretrain_end:world_model_end]
    }
    
    reward_model_data = {
        'states': states[world_model_end:],
        'actions': actions[world_model_end:],
        'rewards': rewards[world_model_end:],
        'stop_episodes': stop_episodes[world_model_end:]
    }
    
    print(f"Split into buckets: pretrain={len(pretrain_data['states'])}, "
          f"world_model={len(world_model_data['states'])}, "
          f"reward_model={len(reward_model_data['states'])}")
    
    # Create datasets for each bucket
    os.makedirs(output_dir, exist_ok=True)
    
    for bucket_name, bucket_data in [
        ("pretrain", pretrain_data),
        ("world_model", world_model_data),
        ("reward_model", reward_model_data)
    ]:
        print(f"Creating {bucket_name} dataset...")
        
        # Split into train/validation
        n_total = len(bucket_data['states'])
        n_train = int(n_total * (1 - validation_split))
        
        train_data = {
            'states': bucket_data['states'][:n_train],
            'actions': bucket_data['actions'][:n_train],
            'rewards': bucket_data['rewards'][:n_train],
            'stop_episodes': bucket_data['stop_episodes'][:n_train]
        }
        
        val_data = {
            'states': bucket_data['states'][n_train:],
            'actions': bucket_data['actions'][n_train:],
            'rewards': bucket_data['rewards'][n_train:],
            'stop_episodes': bucket_data['stop_episodes'][n_train:]
        }
        
        # Create ExperienceDataset objects
        train_dataset = ExperienceDataset(
            states=train_data['states'],
            actions=train_data['actions'],
            rewards=train_data['rewards'],
            stop_episodes=train_data['stop_episodes'],
            sequence_length=sequence_length
        )
        
        val_dataset = ExperienceDataset(
            states=val_data['states'],
            actions=val_data['actions'],
            rewards=val_data['rewards'],
            stop_episodes=val_data['stop_episodes'],
            sequence_length=sequence_length
        )
        
        # Create metadata
        metadata = {
            'environment_name': 'ALE/Breakout-v5',
            'collection_method': 'rlu_converted',
            'bucket': bucket_name,
            'image_height': 84,
            'image_width': 84,
            'validation_split_ratio': validation_split,
            'num_train_transitions': len(train_dataset),
            'num_val_transitions': len(val_dataset),
            'sequence_length': sequence_length,
            'source_dataset': 'RLU_Atari_150k',
            'conversion_date': str(Path(__file__).stat().st_mtime)
        }
        
        # Save dataset
        output_path = os.path.join(output_dir, f"{base_filename}_{bucket_name}.pth")
        payload = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'metadata': metadata
        }
        
        torch.save(payload, output_path)
        print(f"Saved {bucket_name} dataset to {output_path}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")


def flatten_episodes(episodes: List[List[Tuple]]) -> Tuple[List, List, List, List]:
    """
    Flatten episodes into states, actions, rewards, stop_episodes lists.
    
    Args:
        episodes: List of episodes, each episode is a list of transitions
        
    Returns:
        Tuple of (states, actions, rewards, stop_episodes)
    """
    states, actions, rewards, stop_episodes = [], [], [], []
    
    for episode in episodes:
        for transition in episode:
            obs, action, reward, next_obs, done = transition
            states.append(obs)
            actions.append(torch.tensor(action, dtype=torch.int32))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            stop_episodes.append(bool(done))
    
    return states, actions, rewards, stop_episodes


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RLU Atari data to LWM format")
    parser.add_argument("--input", required=True, help="Path to RLU data file (.pkl or .npz)")
    parser.add_argument("--output-dir", default="datasets", help="Output directory")
    parser.add_argument("--base-filename", default="atari_breakout_150k", help="Base filename for output")
    parser.add_argument("--sequence-length", type=int, default=6, help="Sequence length for ExperienceDataset")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    convert_rlu_data_to_lwm_format(
        rlu_data_path=args.input,
        output_dir=args.output_dir,
        base_filename=args.base_filename,
        sequence_length=args.sequence_length,
        validation_split=args.validation_split
    )
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()