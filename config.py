"""
config.py — Central configuration for the Cooperative Spectrum Sensing Simulation.
All tunable hyperparameters and constants are defined here.
"""

import numpy as np
import os

# ──────────────────────────── Paths ────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────── Signal Parameters ────────────────────────────
SAMPLE_RATE = 1_000_000          # 1 MHz baseband sampling rate
NUM_SAMPLES = 1024               # IQ samples per frame
CARRIER_FREQ = 100_000           # 100 kHz carrier frequency
MESSAGE_FREQ = 1_000             # 1 kHz message signal for AM/FM

# Modulation-specific
AM_MODULATION_INDEX = 0.8        # AM modulation depth
FM_FREQ_DEVIATION = 75_000       # FM frequency deviation (Hz)
SYMBOL_RATE = 50_000             # Symbol rate for BPSK/QPSK (symbols/sec)
ROLLOFF = 0.35                   # Raised cosine roll-off factor

# ──────────────────────────── Dataset Parameters ────────────────────────────
CLASSES = ["AM", "FM", "BPSK", "QPSK"]
NUM_CLASSES = len(CLASSES)
SNR_RANGE_DB = np.arange(-20, 22, 4)   # -20 dB to +20 dB, step 4 (11 SNR levels instead of 21)
SAMPLES_PER_CLASS_PER_SNR = 35          # 35 samples per class (1,540 total vs 16,800)
RANDOM_SEED = 42

# ──────────────────────────── Spectrogram Parameters ────────────────────────────
STFT_NPERSEG = 64                # STFT window length
STFT_NOVERLAP = 48               # STFT overlap
STFT_WINDOW = "hann"             # Window function
SPEC_SIZE = (64, 64)             # 64x64 (4x fewer pixels = 4x faster convolutions!)
EPSILON = 1e-12                  # Numerical stability for log

# ──────────────────────────── CNN / Training Parameters ────────────────────────────
BATCH_SIZE = 64                  # Larger batch size for faster vectorized CPU computation
EPOCHS = 10                      # 10 epochs is plenty to reach >90% accuracy
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 4
LR_REDUCE_PATIENCE = 2
LR_REDUCE_FACTOR = 0.5
DROPOUT_RATE = 0.5
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ──────────────────────────── Cooperative Sensing ────────────────────────────
NUM_NODES = 3                    # Number of simulated sensing nodes
OCCUPIED_THRESHOLD = 0.6         # Confidence threshold for "Channel Occupied"

# ──────────────────────────── Misc ────────────────────────────
PLOT_DPI = 150
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_LARGE = (14, 8)
