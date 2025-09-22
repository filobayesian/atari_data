# pull_checkpoints_breakout.py
import os, itertools, numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch, torch.nn as nn, torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print(">>> starting (checkpoints variant)")

# --- device detection (CUDA for Vast AI, fallback to CPU) ---
if torch.cpu.is_available():
    device = torch.device("cpu")
    print(">>> Using CPU")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using MPS (Apple Silicon)")
else:
    device = torch.device("cuda")
    print(f">>> CUDA device: {torch.cuda.get_device_name(0)}")
    print(f">>> CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(">>> device:", device)

# --- config: single checkpoint for Vast AI deployment ---
GAME = "Breakout"
RUN = 5
CHECKPOINT_ID = 45  # Single checkpoint - adjust as needed
CKPTS = [CHECKPOINT_ID]
SPLIT = f"checkpoint_{CHECKPOINT_ID:02d}"
EPISODE_RETURN_MIN = None     # e.g., 5.0 to filter weak episodes; None = no filter
BATCH = 128
print(">>> split:", SPLIT)
print(">>> checkpoint:", CHECKPOINT_ID)

# --- load TFDS (checkpoints are ordered; rewards clipped; obs: 84x84x1) ---
# Using try_gcs=True to fetch pre-prepared data from GCS if available.
try:
    print(">>> Loading dataset...")
    ds = tfds.load(
        f"rlu_atari_checkpoints/{GAME}_run_{RUN}",
        split=SPLIT,
        try_gcs=True,
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=0),
    )
    print(">>> Dataset loaded successfully")
    print(">>> tensorflow:", tf.__version__)
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    print(">>> Available datasets:")
    try:
        available_ds = tfds.list_builders()
        atari_ds = [ds for ds in available_ds if 'atari' in ds.lower()]
        for ds_name in atari_ds[:10]:  # Show first 10 Atari datasets
            print(f"  - {ds_name}")
    except:
        print("  - Could not list available datasets")
    raise

# --- helper: convert one episode -> stream of 4-frame transitions ---
def episode_to_transitions_numpy(ep):
    # ep has: checkpoint_id, episode_id, episode_return, steps (a nested dataset)
    if (EPISODE_RETURN_MIN is not None) and (float(ep["episode_return"]) < EPISODE_RETURN_MIN):
        return
    steps_np = list(tfds.as_numpy(ep["steps"]))
    T = len(steps_np)
    if T < 5:  # need at least 5 frames to build (s_t, s_{t+1}) with 4-frame stacks
        return
    # extract arrays
    obs = np.stack([s["observation"][:, :, 0] for s in steps_np], axis=0)  # (T,84,84), uint8
    act = np.asarray([s["action"] for s in steps_np], dtype=np.int64)      # (T,)
    rew = np.asarray([s["reward"] for s in steps_np], dtype=np.float32)    # clipped [-1,1]
    disc = np.asarray([s["discount"] for s in steps_np], dtype=np.float32) # 0.0 => terminal
    done = (disc == 0.0).astype(np.float32)
    # build transitions aligned at time t (use next-step reward/done)
    for t in range(3, T - 1):
        s_t   = obs[t-3:t+1]        # (4,84,84)
        s_tp1 = obs[t-2:t+2]        # (4,84,84)
        a_t   = act[t]
        r_tp1 = rew[t+1]
        d_tp1 = done[t+1]
        yield (s_t, a_t, r_tp1, s_tp1, d_tp1)

def transitions_generator(ds):
    for ep in tfds.as_numpy(ds):  # iterate episodes
        yield from episode_to_transitions_numpy(ep)

gen = transitions_generator(ds)

# --- get one batch and move to torch (B,4,84,84) ---
def take_batch(g, batch_size):
    buf = list(itertools.islice(g, batch_size))
    if not buf:
        raise RuntimeError("Dataset empty or too strict EPISODE_RETURN_MIN.")
    
    try:
        s  = torch.from_numpy(np.stack([b[0] for b in buf], axis=0)).to(device).float() / 255.0
        a  = torch.from_numpy(np.array([b[1] for b in buf], dtype=np.int64)).to(device)
        r  = torch.from_numpy(np.array([b[2] for b in buf], dtype=np.float32)).to(device)
        sp = torch.from_numpy(np.stack([b[3] for b in buf], axis=0)).to(device).float() / 255.0
        d  = torch.from_numpy(np.array([b[4] for b in buf], dtype=np.float32)).to(device)
        return s, a, r, sp, d
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise

try:
    print(">>> Processing data...")
    o_t, a_t, r_tp1, o_tp1, done = take_batch(gen, BATCH)
    print(">>> batch shapes:", o_t.shape, a_t.shape, r_tp1.shape, o_tp1.shape, done.shape)
    print(">>> data types:", o_t.dtype, a_t.dtype, r_tp1.dtype, o_tp1.dtype, done.dtype)
except Exception as e:
    logger.error(f"Failed to process batch: {e}")
    raise

# --- tiny DQN step (sanity check) ---
class QNet(nn.Module):
    def __init__(self, in_ch=4, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),    nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),    nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    def forward(self, x): return self.head(self.conv(x))

# --- DQN training with error handling ---
try:
    n_actions = int(a_t.max().item()) + 1 if a_t.numel() else 4
    print(f">>> Number of actions: {n_actions}")
    
    # Create networks
    net = QNet(o_t.shape[1], n_actions).to(device)
    tgt = QNet(o_t.shape[1], n_actions).to(device)
    tgt.load_state_dict(net.state_dict())
    
    # Optimizer and hyperparameters
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    gamma = 0.99
    
    print(">>> Networks created successfully")
    
    # Training step
    with torch.no_grad():
        q_next = tgt(o_tp1).amax(1)
        target_q = r_tp1 + (1.0 - done) * gamma * q_next

    q = net(o_t)
    q_sa = q.gather(1, a_t.view(-1,1)).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, target_q)
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    
    print(">>> DQN step loss:", float(loss.item()))
    
    # Memory cleanup
    if torch.cuda.is_available():
        print(f">>> GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f">>> GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print(">>> Training completed successfully")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise
finally:
    # Cleanup
    if 'net' in locals():
        del net
    if 'tgt' in locals():
        del tgt
    if 'opt' in locals():
        del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(">>> done")