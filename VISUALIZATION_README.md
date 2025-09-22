# Latent World Models Data Visualization

This directory contains comprehensive visualization tools for the latent world models dataset files (.pth) containing Atari Breakout data.

## 📁 Files

- `src/utils/visualize_latent_world_models.py` - Main visualization script
- `src/utils/example_visualization.py` - Usage examples
- `src/utils/extract_latent_data.py` - Data extraction utilities (for debugging)

## 🚀 Quick Start

### Command Line Usage

```bash
# Visualize a specific sequence from pretrain bucket
python src/utils/visualize_latent_world_models.py --bucket pretrain --sequence_idx 0

# Create comprehensive dataset summary
python src/utils/visualize_latent_world_models.py

# Save sample frames as images
python src/utils/visualize_latent_world_models.py --save_samples --num_samples 5

# Visualize specific bucket with all plots
python src/utils/visualize_latent_world_models.py --bucket world_model --save_samples
```

### Python API Usage

```python
from src.utils.visualize_latent_world_models import LatentWorldModelsVisualizer

# Initialize visualizer
visualizer = LatentWorldModelsVisualizer("latent_world_models_data")

# Load datasets
datasets = visualizer.load_dataset("atari_breakout_150k")

# Visualize a sequence
visualizer.visualize_sequence("pretrain", sequence_idx=0, split="train")

# Plot statistics
visualizer.plot_reward_statistics("pretrain")
visualizer.plot_action_statistics("pretrain")

# Create comprehensive summary
visualizer.create_dataset_summary()
```

## 🎨 Visualization Features

### 1. Sequence Visualization
- Shows 6-timestep sequences as 2x3 grid
- Displays actions, rewards, and episode boundaries
- Perfect for understanding game dynamics

### 2. Frame Differences
- Shows motion/change between consecutive frames
- Helps identify ball movement and game state changes
- Uses color-coded difference maps

### 3. Reward Statistics
- Reward distribution histograms
- Positive vs negative reward analysis
- Reward trends over time
- Statistical summaries

### 4. Action Statistics
- Action distribution across all actions (NOOP, FIRE, UP, DOWN)
- Action transition probabilities
- Action sequences over time
- Padding action analysis

### 5. Dataset Summary
- Comprehensive overview of all buckets
- Dataset size comparisons
- Reward percentage analysis
- Metadata summaries

### 6. Sample Frame Export
- Save individual sequences as PNG images
- 2x3 grid layout showing all 6 timesteps
- Includes metadata files with sequence information

## 📊 Available Datasets

The visualization works with three dataset buckets:

1. **pretrain** - Pretraining bucket (39,800 train, 9,950 val sequences)
2. **world_model** - World model training bucket (39,800 train, 9,950 val sequences)  
3. **reward_model** - Reward model training bucket (39,800 train, 9,950 val sequences)

Each bucket contains:
- **States**: [N, 84, 84] float32, normalized [0,1] range
- **Actions**: [N] int32, discrete actions (0-3)
- **Rewards**: [N] float32, environment rewards
- **Stop episodes**: [N] bool, episode boundary markers

## 🎮 Action Mapping

- **0**: NOOP (No operation)
- **1**: FIRE (Start game)
- **2**: UP (Move paddle up)
- **3**: DOWN (Move paddle down)

## 📈 Key Statistics

- **Total frames per bucket**: 50,050 transitions (40,040 train + 10,010 val)
- **Sequence length**: 6 timesteps
- **Image size**: 84x84 pixels
- **Positive reward density**: ~3% (optimized for maximum positive rewards)
- **Episode boundaries**: 40 episodes per bucket (1000 transitions each)

## 🔧 Command Line Options

```bash
--data_dir DATA_DIR          Directory containing .pth files (default: latent_world_models_data)
--base_name BASE_NAME        Base name of dataset files (default: atari_breakout_150k)
--bucket {pretrain,world_model,reward_model}  Specific bucket to visualize
--sequence_idx SEQUENCE_IDX  Sequence index to visualize (default: 0)
--split {train,val}          Dataset split to use (default: train)
--output_dir OUTPUT_DIR      Output directory for saved images (default: visualization_output)
--save_samples               Save sample frames as images
--num_samples NUM_SAMPLES    Number of sample sequences to save (default: 5)
```

## 📝 Output Files

The visualization generates various output files:

- `{bucket}_sequence_{idx}.png` - Sequence visualization
- `{bucket}_differences_{idx}.png` - Frame differences
- `{bucket}_rewards.png` - Reward statistics
- `{bucket}_actions.png` - Action statistics
- `dataset_summary.png` - Comprehensive summary
- `{bucket}_samples/` - Directory with sample frame images and metadata

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'seaborn'**
   - This is not critical - the script will use matplotlib defaults
   - Install seaborn with `pip install seaborn` for better styling

2. **Can't get attribute 'ExperienceDataset'**
   - The script includes the ExperienceDataset class definition
   - Make sure you're running from the correct directory

3. **File not found errors**
   - Ensure the .pth files exist in the specified data directory
   - Check the base_name parameter matches your file naming

### Dependencies

Required packages:
- torch
- numpy
- matplotlib
- PIL (Pillow)

Optional packages:
- seaborn (for better plot styling)

## 🎯 Use Cases

This visualization tool is perfect for:

- **Data Analysis**: Understanding the structure and content of the dataset
- **Model Debugging**: Visualizing sequences to debug model behavior
- **Research**: Analyzing game dynamics and reward patterns
- **Documentation**: Creating visual documentation of the dataset
- **Quality Assurance**: Verifying data integrity and format correctness

## 📚 Examples

See `src/utils/example_visualization.py` for complete usage examples demonstrating all features of the visualization system.
