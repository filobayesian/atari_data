# read_local_shard_fixed.py
import os, numpy as np

print(">>> starting script")

# ---- device (MPS) ----
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(">>> device:", device)

# ---- point to your local shard ----
DATA_DIR = os.path.expanduser("~/rlu_shards/Breakout")   # change if different
SHARD = os.path.join(DATA_DIR, "run_5-00000-of-00001")   # the file you copied
if not os.path.exists(SHARD):
    raise FileNotFoundError(f"Shard not found: {SHARD}")
print(">>> shard:", SHARD, "| size:", round(os.path.getsize(SHARD)/(1024**3),3), "GiB")

# ---- TensorFlow (no TFDS/Beam) ----
import tensorflow as tf
print(">>> tensorflow:", tf.__version__)

BATCH = 128
H, W = 84, 84
COMPRESSION = "GZIP"  # your file header 0x1f8b confirms gzip

# ----------------- PNG-based transition parser -----------------
def _decode_obs_png(var_sparse):
    """Accepts VarLenFeature(tf.string) for PNG-encoded observations; returns [84,84,4] uint8.
       Decodes 4 separate PNG images and stacks them along the channel dimension."""
    var_dense = tf.sparse.to_dense(var_sparse)            # shape [4] - 4 PNG images
    k = tf.shape(var_dense)[0]
    
    # Decode each PNG image
    def decode_png(png_bytes):
        # Decode PNG to get raw pixel data
        png_decoded = tf.io.decode_png(png_bytes, channels=1)  # Grayscale PNG
        return tf.cast(png_decoded, tf.uint8)  # [84, 84, 1]
    
    # Decode all 4 PNG images
    decoded_frames = tf.map_fn(
        decode_png,
        var_dense,
        fn_output_signature=tf.uint8,
    )  # [4, 84, 84, 1]
    
    # Stack along channel dimension: [4, 84, 84, 1] -> [84, 84, 4]
    stacked = tf.transpose(decoded_frames, [1, 2, 0, 3])  # [84, 84, 4, 1]
    return tf.squeeze(stacked, axis=3)  # [84, 84, 4]

def parse_transition_png(rec):
    features = {
        "o_t":   tf.io.VarLenFeature(tf.string),   # 4 PNG images
        "a_t":   tf.io.FixedLenFeature([], tf.int64),
        "r_t":   tf.io.FixedLenFeature([], tf.float32),
        "d_t":   tf.io.FixedLenFeature([], tf.float32),  # discount: 0.0 => terminal
        "o_tp1": tf.io.VarLenFeature(tf.string),   # 4 PNG images
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
    }
    p = tf.io.parse_single_example(rec, features)
    o_t   = _decode_obs_png(p["o_t"])                    # [84,84,4]
    o_tp1 = _decode_obs_png(p["o_tp1"])                  # [84,84,4]
    a_t   = tf.cast(p["a_t"], tf.int32)
    r_tp1 = p["r_t"]
    done  = tf.cast(tf.equal(p["d_t"], 0.0), tf.float32)    # terminal if discount==0
    return o_t, a_t, r_tp1, o_tp1, done

# Build dataset (single-thread = macOS-safe)
ds = tf.data.TFRecordDataset(
    [SHARD],
    compression_type=COMPRESSION,
    num_parallel_reads=1,
).map(parse_transition_png, num_parallel_calls=1).prefetch(1)

print(">>> materializing one batch")
o_t, a_t, r_tp1, o_tp1, done = next(iter(ds.batch(BATCH)))

# ----------------- Torch + MPS: convert & one DQN step -----------------
import torch.nn as nn
import torch.nn.functional as F

# to torch tensors
o_t   = torch.from_numpy(o_t.numpy()).to(device=device, dtype=torch.float32) / 255.0
o_tp1 = torch.from_numpy(o_tp1.numpy()).to(device=device, dtype=torch.float32) / 255.0
a_t   = torch.from_numpy(a_t.numpy()).to(device=device, dtype=torch.long)
r_tp1 = torch.from_numpy(r_tp1.numpy()).to(device=device, dtype=torch.float32)
done  = torch.from_numpy(done.numpy()).to(device=device, dtype=torch.float32)

# (B, H, W, C) -> (B, C, H, W)
o_t   = o_t.permute(0, 3, 1, 2)
o_tp1 = o_tp1.permute(0, 3, 1, 2)
in_ch = o_t.shape[1]
print(">>> batch shapes:", o_t.shape, a_t.shape, r_tp1.shape, o_tp1.shape, done.shape)

# tiny DQN net
class QNet(nn.Module):
    def __init__(self, in_ch=4, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    def forward(self, x): return self.head(self.conv(x))

n_actions = int(a_t.max().item()) + 1 if a_t.numel() else 4
online, target = QNet(in_ch, n_actions).to(device), QNet(in_ch, n_actions).to(device)
target.load_state_dict(online.state_dict())
opt, gamma = torch.optim.Adam(online.parameters(), lr=1e-4), 0.99

with torch.no_grad():
    q_next = target(o_tp1).amax(1)
    target_q = r_tp1 + (1.0 - done) * gamma * q_next

q = online(o_t)
q_sa = q.gather(1, a_t.view(-1,1)).squeeze(1)
loss = F.smooth_l1_loss(q_sa, target_q)
opt.zero_grad(set_to_none=True)
loss.backward()
nn.utils.clip_grad_norm_(online.parameters(), 10.0)
opt.step()

print(">>> DQN step loss:", float(loss.item()))
print(">>> done")
