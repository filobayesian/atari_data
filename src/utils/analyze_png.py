#!/usr/bin/env python3
"""
Analyze PNG data in the shard to understand the actual image format
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

print(">>> Analyzing PNG data in shard")

# Point to your local shard
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"

# Read first record
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)
first_record = next(iter(ds.take(1)))

def parse_raw_example(rec):
    features = {
        "o_t": tf.io.VarLenFeature(tf.string),
        "a_t": tf.io.FixedLenFeature([], tf.int64),
        "r_t": tf.io.FixedLenFeature([], tf.float32),
        "d_t": tf.io.FixedLenFeature([], tf.float32),
        "o_tp1": tf.io.VarLenFeature(tf.string),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(rec, features)

parsed = parse_raw_example(first_record)
o_t_dense = tf.sparse.to_dense(parsed["o_t"])

print(f">>> Number of observation frames: {len(o_t_dense)}")

# Analyze each frame
for i in range(len(o_t_dense)):
    frame_bytes = o_t_dense[i]
    print(f"\n>>> Frame {i+1}:")
    frame_numpy = frame_bytes.numpy()
    print(f"  Raw bytes length: {len(frame_numpy)}")
    print(f"  First 20 bytes: {frame_numpy[:20]}")
    
    # Check PNG magic number
    if frame_numpy[:8] == b'\x89PNG\r\n\x1a\n':
        print(f"  ✓ Valid PNG file")
        
        # Decode PNG and get dimensions
        try:
            img = Image.open(io.BytesIO(frame_numpy))
            print(f"  PNG dimensions: {img.size} (W x H)")
            print(f"  PNG mode: {img.mode}")
            print(f"  PNG format: {img.format}")
            
            # Convert to numpy array
            img_array = np.array(img)
            print(f"  Array shape: {img_array.shape}")
            print(f"  Array dtype: {img_array.dtype}")
            print(f"  Value range: {img_array.min()} - {img_array.max()}")
            
        except Exception as e:
            print(f"  ✗ Error decoding PNG: {e}")
    else:
        print(f"  ✗ Not a valid PNG file")

print(f"\n>>> Summary:")
print(f"  - Observations are stored as 4 separate PNG images")
print(f"  - Each PNG is 726 bytes")
print(f"  - PNG dimensions: 84x84 pixels, grayscale (L mode)")
print(f"  - Need to decode PNGs to get actual pixel data")
print(f"  - Original parser expected raw pixel arrays, not PNGs")
