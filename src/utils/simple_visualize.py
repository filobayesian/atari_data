#!/usr/bin/env python3
"""
Simple visualization that saves images to files
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image

print(">>> Simple visualization - saving images to files")

# ---- Setup ----
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"

# Create output directory
output_dir = "visualization_output"
os.makedirs(output_dir, exist_ok=True)

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

# ---- Load and visualize data ----
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
ds = ds.map(parse_transition_png, num_parallel_calls=1).prefetch(1)

# Get a few samples
samples = list(ds.take(3))
print(f">>> Loaded {len(samples)} samples")

for i, (o_t, a_t, r_tp1, o_tp1, done) in enumerate(samples):
    print(f"\n>>> Processing sample {i+1}:")
    print(f"  Action: {a_t.numpy()}")
    print(f"  Reward: {r_tp1.numpy():.2f}")
    print(f"  Done: {bool(done.numpy())}")
    
    # Convert to numpy
    o_t_np = o_t.numpy()
    o_tp1_np = o_tp1.numpy()
    
    # Save individual frames
    for frame_idx in range(4):
        # Current observation frames
        frame = o_t_np[:, :, frame_idx]
        img = Image.fromarray(frame, mode='L')
        img.save(f"{output_dir}/sample_{i+1}_o_t_frame_{frame_idx+1}.png")
        
        # Next observation frames
        frame = o_tp1_np[:, :, frame_idx]
        img = Image.fromarray(frame, mode='L')
        img.save(f"{output_dir}/sample_{i+1}_o_tp1_frame_{frame_idx+1}.png")
    
    # Create a combined image showing all 4 frames side by side
    def create_combined_image(frames, title):
        # Create a 2x2 grid of frames
        combined = np.zeros((84*2, 84*2), dtype=np.uint8)
        combined[0:84, 0:84] = frames[:, :, 0]
        combined[0:84, 84:168] = frames[:, :, 1]
        combined[84:168, 0:84] = frames[:, :, 2]
        combined[84:168, 84:168] = frames[:, :, 3]
        return combined
    
    # Save combined images
    o_t_combined = create_combined_image(o_t_np, f"Sample {i+1} o_t")
    img = Image.fromarray(o_t_combined, mode='L')
    img.save(f"{output_dir}/sample_{i+1}_o_t_combined.png")
    
    o_tp1_combined = create_combined_image(o_tp1_np, f"Sample {i+1} o_tp1")
    img = Image.fromarray(o_tp1_combined, mode='L')
    img.save(f"{output_dir}/sample_{i+1}_o_tp1_combined.png")
    
    # Create frame difference
    diff = np.abs(o_tp1_np.astype(np.float32) - o_t_np.astype(np.float32)).astype(np.uint8)
    diff_combined = create_combined_image(diff, f"Sample {i+1} difference")
    img = Image.fromarray(diff_combined, mode='L')
    img.save(f"{output_dir}/sample_{i+1}_difference.png")
    
    print(f"  Saved images to {output_dir}/")

print(f"\n>>> All images saved to {output_dir}/ directory")
print(">>> Files created:")
print("  - sample_X_o_t_frame_Y.png: Individual frames from current observation")
print("  - sample_X_o_tp1_frame_Y.png: Individual frames from next observation") 
print("  - sample_X_o_t_combined.png: 2x2 grid of current observation frames")
print("  - sample_X_o_tp1_combined.png: 2x2 grid of next observation frames")
print("  - sample_X_difference.png: Frame differences showing motion")
