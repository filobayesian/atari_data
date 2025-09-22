#!/usr/bin/env python3
"""
Extract raw data from latent world models .pth files
This script handles the ExperienceDataset class loading issue by extracting the raw tensors.
"""

import os
import torch
import numpy as np
import pickle
from typing import Dict, Any


def extract_raw_data_from_pth(filename: str) -> Dict[str, Any]:
    """
    Extract raw data from a .pth file, handling the ExperienceDataset class issue
    
    Args:
        filename: Path to the .pth file
        
    Returns:
        Dictionary containing the raw data
    """
    print(f"Extracting data from {filename}...")
    
    try:
        # Method 1: Try torch.load with custom unpickler
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ExperienceDataset':
                    # Return a simple class that can hold the data
                    class SimpleExperienceDataset:
                        def __init__(self, states, actions, rewards, stop_episodes, sequence_length=6):
                            self.states = states
                            self.actions = actions
                            self.rewards = rewards
                            self.stop_episodes = stop_episodes
                            self.sequence_length = sequence_length
                            
                        def __len__(self):
                            # Calculate valid sequences
                            valid_indices = []
                            N = len(self.stop_episodes)
                            for start in range(N - self.sequence_length):
                                window = self.stop_episodes[start : start + self.sequence_length]
                                if not window.any():
                                    valid_indices.append(start)
                            return len(valid_indices)
                            
                        def __getitem__(self, idx):
                            # Calculate valid start indices
                            valid_indices = []
                            N = len(self.stop_episodes)
                            for start in range(N - self.sequence_length):
                                window = self.stop_episodes[start : start + self.sequence_length]
                                if not window.any():
                                    valid_indices.append(start)
                            
                            start_idx = valid_indices[idx]
                            end_idx = start_idx + self.sequence_length
                            
                            return {
                                'states': self.states[start_idx:end_idx],
                                'actions': self.actions[start_idx:end_idx],
                                'rewards': self.rewards[start_idx:end_idx],
                                'stop_episodes': self.stop_episodes[start_idx:end_idx]
                            }
                    
                    return SimpleExperienceDataset
                return super().find_class(module, name)
        
        with open(filename, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
        
        print(f"  ✅ Successfully loaded with custom unpickler")
        return data
        
    except Exception as e:
        print(f"  ❌ Custom unpickler failed: {e}")
        
        # Method 2: Try to extract raw tensors directly
        try:
            print(f"  🔄 Trying raw tensor extraction...")
            
            # Load the file as a raw pickle to inspect structure
            with open(filename, 'rb') as f:
                raw_data = pickle.load(f)
            
            print(f"  Raw data type: {type(raw_data)}")
            print(f"  Raw data keys: {list(raw_data.keys()) if hasattr(raw_data, 'keys') else 'No keys'}")
            
            # If it's a dict, try to extract the datasets
            if isinstance(raw_data, dict):
                extracted_data = {}
                for key, value in raw_data.items():
                    if hasattr(value, 'states') and hasattr(value, 'actions'):
                        # This is likely an ExperienceDataset
                        extracted_data[key] = {
                            'states': value.states,
                            'actions': value.actions,
                            'rewards': value.rewards,
                            'stop_episodes': value.stop_episodes,
                            'sequence_length': getattr(value, 'sequence_length', 6)
                        }
                    else:
                        extracted_data[key] = value
                
                print(f"  ✅ Extracted raw tensors")
                return extracted_data
            else:
                print(f"  ❌ Unexpected data structure: {type(raw_data)}")
                return None
                
        except Exception as e2:
            print(f"  ❌ Raw extraction also failed: {e2}")
            return None


def load_all_datasets(data_dir: str = "latent_world_models_data", 
                     base_name: str = "atari_breakout_150k") -> Dict[str, Any]:
    """
    Load all three dataset files
    
    Args:
        data_dir: Directory containing .pth files
        base_name: Base name of dataset files
        
    Returns:
        Dictionary containing all loaded datasets
    """
    buckets = ['pretrain', 'world_model', 'reward_model']
    all_data = {}
    
    print(f"Loading all datasets: {base_name}")
    print("=" * 50)
    
    for bucket in buckets:
        filename = os.path.join(data_dir, f"{base_name}_{bucket}.pth")
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
            
        data = extract_raw_data_from_pth(filename)
        if data is not None:
            all_data[bucket] = data
            print(f"✅ Loaded {bucket}")
        else:
            print(f"❌ Failed to load {bucket}")
    
    return all_data


if __name__ == "__main__":
    # Test the extraction
    data = load_all_datasets()
    
    if data:
        print(f"\n📊 Successfully loaded {len(data)} datasets")
        for bucket, bucket_data in data.items():
            print(f"\n{bucket.upper()}:")
            for key, value in bucket_data.items():
                if isinstance(value, dict) and 'states' in value:
                    print(f"  {key}: {value['states'].shape} states, {value['actions'].shape} actions")
                else:
                    print(f"  {key}: {type(value)}")
    else:
        print("❌ No data loaded successfully")
