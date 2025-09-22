#!/usr/bin/env python3
"""
Debug script to examine the actual format of the Atari shard data
"""
import os
import tensorflow as tf
import numpy as np

print(">>> Debugging shard format")

# Point to your local shard
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")
COMPRESSION = "GZIP"

if not os.path.exists(SHARD):
    raise FileNotFoundError(f"Shard not found: {SHARD}")

print(f">>> Shard: {SHARD}")
print(f">>> Size: {os.path.getsize(SHARD)/(1024**3):.3f} GiB")

# Read raw records without parsing
ds = tf.data.TFRecordDataset([SHARD], compression_type=COMPRESSION, num_parallel_reads=1)

print("\n>>> Examining first few records...")

# Get first record
first_record = next(iter(ds.take(1)))
print(f">>> First record type: {type(first_record)}")
print(f">>> First record shape: {first_record.shape}")

# Parse as raw example to see feature structure
def parse_raw_example(rec):
    """Parse example to see what features are actually present"""
    features = {
        "o_t": tf.io.VarLenFeature(tf.string),
        "a_t": tf.io.FixedLenFeature([], tf.int64),
        "r_t": tf.io.FixedLenFeature([], tf.float32),
        "d_t": tf.io.FixedLenFeature([], tf.float32),
        "o_tp1": tf.io.VarLenFeature(tf.string),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(rec, features)

# Parse first record
parsed = parse_raw_example(first_record)
print(f"\n>>> Parsed features:")
for key, value in parsed.items():
    if hasattr(value, 'values'):
        print(f"  {key}: VarLen with {tf.shape(value.values)[0]} values")
        print(f"    Values shape: {value.values.shape}")
        print(f"    Values dtype: {value.values.dtype}")
    else:
        print(f"  {key}: {value} (shape: {value.shape}, dtype: {value.dtype})")

# Examine observation data in detail
print(f"\n>>> Detailed observation analysis:")
o_t_sparse = parsed["o_t"]
o_tp1_sparse = parsed["o_tp1"]

print(f"o_t sparse shape: {o_t_sparse.values.shape}")
print(f"o_tp1 sparse shape: {o_tp1_sparse.values.shape}")

# Convert to dense and examine
o_t_dense = tf.sparse.to_dense(o_t_sparse)
o_tp1_dense = tf.sparse.to_dense(o_tp1_sparse)

print(f"o_t dense shape: {o_t_dense.shape}")
print(f"o_tp1 dense shape: {o_tp1_dense.shape}")

# Decode first observation
print(f"\n>>> Decoding first observation:")
raw_o_t = tf.io.decode_raw(o_t_dense[0], tf.uint8)
print(f"Raw o_t shape after decode_raw: {raw_o_t.shape}")
print(f"Raw o_t values: {raw_o_t.numpy()[:20]}...")  # First 20 values

# Try different reshape possibilities
print(f"\n>>> Testing different reshape possibilities:")
print(f"Raw data length: {len(raw_o_t)}")

# Common Atari frame dimensions to try
possible_shapes = [
    (84, 84, 1),   # Single frame
    (84, 84, 2),   # Two frames
    (84, 84, 3),   # Three frames  
    (84, 84, 4),   # Four frames
    (42, 42, 4),   # Smaller frames
    (64, 64, 1),   # Different size
    (80, 80, 1),   # Different size
]

for h, w, c in possible_shapes:
    total = h * w * c
    if total == len(raw_o_t):
        print(f"  MATCH: {h}x{w}x{c} = {total}")
    else:
        print(f"  No match: {h}x{w}x{c} = {total} (need {len(raw_o_t)})")

# Check if it's a perfect square or has other patterns
print(f"\n>>> Factor analysis of {len(raw_o_t)}:")
n = len(raw_o_t)
factors = []
for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
        factors.append(i)
        if i != n // i:
            factors.append(n // i)
factors.sort()
print(f"Factors: {factors}")

# Try to find reasonable 2D shapes
print(f"\n>>> Possible 2D shapes:")
for f1 in factors:
    f2 = n // f1
    if f1 <= f2:  # Avoid duplicates
        print(f"  {f1} x {f2} = {n}")

print(f"\n>>> Done debugging")
