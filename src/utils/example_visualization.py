#!/usr/bin/env python3
"""
Example usage of the Latent World Models visualization function
"""

from visualize_latent_world_models import LatentWorldModelsVisualizer


def main():
    """Example usage of the visualization function"""
    
    print("🎨 Latent World Models Visualization Example")
    print("=" * 50)
    
    # Initialize the visualizer
    visualizer = LatentWorldModelsVisualizer("latent_world_models_data")
    
    # Load all datasets
    print("\n📁 Loading datasets...")
    datasets = visualizer.load_dataset("atari_breakout_150k")
    
    if not datasets:
        print("❌ No datasets loaded successfully")
        return
    
    print(f"✅ Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Example 1: Visualize a specific sequence
    print("\n🎬 Example 1: Visualizing a sequence")
    visualizer.visualize_sequence(
        bucket="pretrain",
        sequence_idx=0,
        split="train",
        save_path="example_sequence.png"
    )
    
    # Example 2: Show frame differences
    print("\n🔄 Example 2: Frame differences")
    visualizer.visualize_frame_differences(
        bucket="pretrain",
        sequence_idx=0,
        split="train",
        save_path="example_differences.png"
    )
    
    # Example 3: Plot reward statistics
    print("\n📊 Example 3: Reward statistics")
    visualizer.plot_reward_statistics(
        bucket="pretrain",
        save_path="example_rewards.png"
    )
    
    # Example 4: Plot action statistics
    print("\n🎮 Example 4: Action statistics")
    visualizer.plot_action_statistics(
        bucket="pretrain",
        save_path="example_actions.png"
    )
    
    # Example 5: Create comprehensive summary
    print("\n📈 Example 5: Dataset summary")
    visualizer.create_dataset_summary(
        save_path="example_summary.png"
    )
    
    # Example 6: Save sample frames
    print("\n💾 Example 6: Save sample frames")
    visualizer.save_sample_frames(
        bucket="pretrain",
        num_samples=3,
        split="train",
        output_dir="example_samples"
    )
    
    print("\n✅ All examples completed!")
    print("Check the current directory for generated visualizations.")


if __name__ == "__main__":
    main()
