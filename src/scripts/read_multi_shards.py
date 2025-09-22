# read_multi_shards.py
import os, glob, numpy as np

print(">>> starting script")

# ---- device (MPS) ----
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(">>> device:", device)

# ---- where shards live ----
DATA_DIR = os.path.expanduser(os.environ.get("DATA_DIR", "~/rlu_shards/Breakout"))
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "run_5-*-of-*")))
if not FILES:
    raise FileNotFoundError(f"No shards found matching run_5-*-of-* in {DATA_DIR}")
print(">>> found shards:", len(FILES))
for f in FILES:
    print("   -", os.path.basename(f))

# ---- TensorFlow (no TFDS/Beam) ----
import tensorflow as tf
print(">>> tensorflow:", tf.__version__)

BATCH = 128
H, W = 84, 84
COMPRESSION = "GZIP"  # these RL Unplugged shards are gzip TFRecords

# ---------- VarLen PNG/JPEG-aware transition parser ----------
def _decode_obs_varlen(var_sparse):
    """Accept VarLenFeature(tf.string) with either:
       - K=1 blob holding an 84x84x4 image, or
       - K=4 blobs each 84x84x1.
       Decodes PNG/JPEG bytes and returns uint8 [84,84,4]."""
    var_dense = tf.sparse.to_dense(var_sparse)   # [K]
    k = tf.shape(var_dense)[0]

    def decode_k1():
        img = tf.io.decode_image(var_dense[0], channels=0, expand_animations=False)
        img = tf.image.resize_with_crop_or_pad(img, H, W)
        img = tf.ensure_shape(img, [H, W, None])
        c = tf.shape(img)[-1]
        def to4_from1(): return tf.tile(img, [1, 1, 4])
        def to4_from3():
            last = img[..., -1:]
            return tf.concat([img, last], axis=-1)
        return tf.case(
            [(tf.equal(c, 1), to4_from1),
             (tf.equal(c, 3), to4_from3)],
            default=lambda: img
        )

    def decode_kN():
        imgs = tf.map_fn(
            lambda b: tf.io.decode_image(b, channels=1, expand_animations=False),
            var_dense, fn_output_signature=tf.uint8
        )                                   # [K, H, W, 1]
        imgs = tf.transpose(imgs, [1, 2, 0, 3])   # [H, W, K, 1]
        imgs = tf.squeeze(imgs, axis=3)           # [H, W, K]
        kk = tf.shape(imgs)[-1]
        def pad_to4():
            last = imgs[..., -1:]
            reps = 4 - kk
            pads = tf.repeat(last, repeats=reps, axis=-1)
            return tf.concat([imgs, pads], axis=-1)
        def trim_to4():
            return imgs[..., :4]
        def ok4():
            return imgs
        return tf.case(
            [(tf.less(kk, 4), pad_to4),
             (tf.greater(kk, 4), trim_to4)],
            default=ok4
        )

    obs = tf.cond(tf.equal(k, 1), decode_k1, decode_kN)
    obs = tf.ensure_shape(obs, [H, W, 4])
    return obs

def parse_transition_varlen(rec):
    features = {
        "o_t":   tf.io.VarLenFeature(tf.string),   # image bytes (1 or 4 blobs)
        "a_t":   tf.io.FixedLenFeature([], tf.int64),
        "r_t":   tf.io.FixedLenFeature([], tf.float32),
        "d_t":   tf.io.FixedLenFeature([], tf.float32),  # discount: 0.0 => terminal
        "o_tp1": tf.io.VarLenFeature(tf.string),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
    }
    p = tf.io.parse_single_example(rec, features)
    o_t   = _decode_obs_varlen(p["o_t"])                    # [84,84,4]
    o_tp1 = _decode_obs_varlen(p["o_tp1"])                  # [84,84,4]
    a_t   = tf.cast(p["a_t"], tf.int32)
    r_tp1 = p["r_t"]
    done  = tf.cast(tf.equal(p["d_t"], 0.0), tf.float32)    # terminal if discount==0
    return o_t, a_t, r_tp1, o_tp1, done

# Build dataset from ALL shards (single-thread = macOS-safe)
ds = tf.data.TFRecordDataset(
    FILES,
    compression_type=COMPRESSION,
    num_parallel_reads=1,
).map(parse_transition_varlen, num_parallel_calls=1).shuffle(50000).prefetch(1)

print(">>> sampling one batch")
o_t, a_t, r_tp1, o_tp1, done = next(iter(ds.batch(BATCH)))

# ---------- Torch + MPS ----------
import torch.nn as nn
import torch.nn.functional as F

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