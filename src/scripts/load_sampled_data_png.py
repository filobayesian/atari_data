#!/usr/bin/env python3
"""
Utility script to load the sampled 150k frames dataset (PNG format)
Includes PNG decoding functionality when needed
"""
import pickle
import numpy as np
import os
import tensorflow as tf

def decode_png_observation(png_bytes_array):
    """
    Decode PNG observation bytes to pixel arrays
    
    Args:
        png_bytes_array: Array of 4 PNG byte strings (shape: (4,))
    
    Returns:
        np.array: Decoded observation with shape (84, 84, 4)
    """
    def decode_single_png(png_bytes):
        png_decoded = tf.io.decode_png(png_bytes, channels=1)
        return tf.cast(png_decoded, tf.uint8)
    
    # Decode each PNG frame
    decoded_frames = []
    for png_bytes in png_bytes_array:
        decoded_frame = decode_single_png(png_bytes).numpy()
        decoded_frames.append(decoded_frame.squeeze())  # Remove channel dimension
    
    # Stack frames along channel dimension: (4, 84, 84) -> (84, 84, 4)
    stacked = np.stack(decoded_frames, axis=2)
    return stacked

def load_sampled_data_png(data_format='pickle'):
    """
    Load the sampled 150k frames dataset (PNG format)
    
    Args:
        data_format (str): 'pickle' or 'numpy' - format to load data in
    
    Returns:
        dict: Dictionary containing the dataset with keys:
            - observations_t_png: PNG byte strings (150k, 4) - object array
            - observations_tp1_png: PNG byte strings (150k, 4) - object array
            - actions: Actions taken (150k,)
            - rewards: Rewards received (150k,)
            - dones: Episode termination flags (150k,)
            - steps: Original step indices (150k,)
    """
    if not os.path.exists("sampled_data"):
        raise FileNotFoundError("sampled_data directory not found. Run sample_optimal_150k.py first.")
    
    if data_format == 'pickle':
        with open("sampled_data/optimal_150k_frames.pkl", "rb") as f:
            data = pickle.load(f)
    elif data_format == 'numpy':
        data = np.load("sampled_data/optimal_150k_frames.npz", allow_pickle=True)
        # Convert to dict for consistency
        data = {key: data[key] for key in data.keys()}
    else:
        raise ValueError("data_format must be 'pickle' or 'numpy'")
    
    return data

def load_sampled_data_decoded(data_format='pickle'):
    """
    Load the sampled 150k frames dataset with decoded observations
    
    Args:
        data_format (str): 'pickle' or 'numpy' - format to load data in
    
    Returns:
        dict: Dictionary containing the dataset with decoded observations:
            - observations_t: Decoded observations (150k, 84, 84, 4) uint8
            - observations_tp1: Decoded observations (150k, 84, 84, 4) uint8
            - actions: Actions taken (150k,)
            - rewards: Rewards received (150k,)
            - dones: Episode termination flags (150k,)
            - steps: Original step indices (150k,)
    """
    data = load_sampled_data_png(data_format)
    
    print(">>> Decoding PNG observations...")
    # Decode observations
    observations_t = []
    observations_tp1 = []
    
    for i in range(len(data['actions'])):
        if i % 10000 == 0:
            print(f"  Decoded {i:,}/{len(data['actions']):,} observations")
        
        obs_t = decode_png_observation(data['observations_t_png'][i])
        obs_tp1 = decode_png_observation(data['observations_tp1_png'][i])
        
        observations_t.append(obs_t)
        observations_tp1.append(obs_tp1)
    
    # Convert to numpy arrays
    data_decoded = {
        'observations_t': np.array(observations_t, dtype=np.uint8),
        'observations_tp1': np.array(observations_tp1, dtype=np.uint8),
        'actions': data['actions'],
        'rewards': data['rewards'],
        'dones': data['dones'],
        'steps': data['steps']
    }
    
    return data_decoded

def load_metadata():
    """Load dataset metadata"""
    with open("sampled_data/metadata.pkl", "rb") as f:
        return pickle.load(f)

def get_data_info():
    """Print information about the sampled dataset"""
    try:
        metadata = load_metadata()
        data = load_sampled_data_png()
        
        print("="*60)
        print("SAMPLED 150K FRAMES DATASET INFO (PNG FORMAT)")
        print("="*60)
        print(f"📊 Dataset Statistics:")
        print(f"  Total frames: {len(data['rewards']):,}")
        print(f"  Start position: {metadata['start_position']:,}")
        print(f"  End position: {metadata['end_position']:,}")
        print(f"  Positive rewards: {metadata['positive_rewards']:,} ({metadata['positive_reward_percentage']:.2f}%)")
        print(f"  Total reward: {metadata['total_reward']:.1f}")
        print(f"  Mean reward: {metadata['mean_reward']:.6f}")
        print(f"  Episodes ended: {metadata['episodes_ended']:,}")
        print(f"  Observation format: {metadata['observation_format']}")
        print(f"  PNG frames per observation: {metadata['observation_png_count']}")
        print(f"  PNG size: {metadata['observation_png_size_bytes']:,} bytes per frame")
        print(f"  Unique actions: {metadata['unique_actions']}")
        
        print(f"\n💾 Available formats:")
        print(f"  - Pickle: sampled_data/optimal_150k_frames.pkl")
        print(f"  - Numpy: sampled_data/optimal_150k_frames.npz")
        
        print(f"\n🎯 Optimization:")
        print(f"  This dataset maximizes positive rewards in a consecutive 150k frame window")
        print(f"  Found {metadata['positive_rewards']:,} positive rewards out of 150,000 total frames")
        print(f"  This represents {metadata['positive_reward_percentage']:.2f}% positive reward density")
        
        print(f"\n🔧 PNG Format:")
        print(f"  ✅ Original PNG structure preserved")
        print(f"  ✅ Use load_sampled_data_decoded() to get pixel arrays")
        print(f"  ✅ Use decode_png_observation() for individual observations")
        
        return data, metadata
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Run sample_optimal_150k.py first to create the dataset")
        return None, None

def create_training_batches_decoded(batch_size=32, shuffle=True):
    """
    Create training batches from the sampled data with decoded observations
    
    Args:
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        list: List of batches, each containing (obs_t, obs_tp1, actions, rewards, dones)
    """
    data = load_sampled_data_decoded()
    
    # Get data
    obs_t = data['observations_t']
    obs_tp1 = data['observations_tp1'] 
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    # Create indices
    indices = np.arange(len(obs_t))
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = {
            'observations_t': obs_t[batch_indices],
            'observations_tp1': obs_tp1[batch_indices],
            'actions': actions[batch_indices],
            'rewards': rewards[batch_indices],
            'dones': dones[batch_indices]
        }
        batches.append(batch)
    
    return batches

if __name__ == "__main__":
    # Print dataset info when run directly
    get_data_info()
