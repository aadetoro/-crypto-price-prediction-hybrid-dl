# ── SEED CONTROL — add this block at the very top of each script,          ──
# ── BEFORE any other imports (numpy, tensorflow, sklearn, etc.)            ──

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"      # forces TensorFlow to use
                                                # deterministic GPU/CPU kernels
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"     # removes cuDNN's non-deterministic
                                                # convolution/LSTM algorithms

import random
import numpy as np

SEED = 42

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Newer TF versions also expose this for extra determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass  # older TF versions don't have this — the env vars above still help
    except ImportError:
        pass  # pipeline script doesn't need TF, only numpy/random

set_global_seed(SEED)
print(f"✓ Global random seed set to {SEED} — results are now reproducible run-to-run")

# ── END SEED CONTROL BLOCK ── your existing imports and code follow below ──
