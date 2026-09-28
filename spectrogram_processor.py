"""
spectrogram_processor.py — STFT-based Spectrogram Generation.
Converts 1D complex IQ signals into 2D spectrogram images suitable for CNN input.
"""

import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom
import config as cfg


def iq_to_spectrogram(iq_signal, nperseg=None, noverlap=None, window=None,
                      target_size=None, fs=None):
    """
    Convert a single complex IQ signal to a normalized spectrogram image.
    
    Pipeline:
        1. STFT using scipy.signal.stft
        2. Power spectral density: 10 * log10(|STFT|^2 + epsilon)
        3. Min-max normalization to [0, 1]
        4. Resize to target dimensions via bicubic interpolation
    
    Parameters:
        iq_signal: 1D complex numpy array.
        nperseg: STFT window length.
        noverlap: STFT overlap samples.
        window: Window function name.
        target_size: Output (height, width) tuple.
        fs: Sampling frequency.
    
    Returns:
        2D numpy array of shape target_size, values in [0, 1].
    """
    nperseg = nperseg or cfg.STFT_NPERSEG
    noverlap = noverlap or cfg.STFT_NOVERLAP
    window = window or cfg.STFT_WINDOW
    target_size = target_size or cfg.SPEC_SIZE
    fs = fs or cfg.SAMPLE_RATE
    
    # Step 1: Compute STFT
    f, t, Zxx = stft(iq_signal, fs=fs, window=window,
                     nperseg=nperseg, noverlap=noverlap)
    
    # Step 2: Power spectral density in dB
    power = np.abs(Zxx) ** 2
    psd_db = 10 * np.log10(power + cfg.EPSILON)
    
    # Step 3: Min-max normalization to [0, 1]
    psd_min = psd_db.min()
    psd_max = psd_db.max()
    if psd_max - psd_min > 0:
        normalized = (psd_db - psd_min) / (psd_max - psd_min)
    else:
        normalized = np.zeros_like(psd_db)
    
    # Step 4: Resize to target dimensions using bicubic interpolation
    zoom_factors = (
        target_size[0] / normalized.shape[0],
        target_size[1] / normalized.shape[1]
    )
    resized = zoom(normalized, zoom_factors, order=3)  # order=3 = bicubic
    
    # Clip to [0, 1] after interpolation (bicubic can overshoot)
    resized = np.clip(resized, 0.0, 1.0)
    
    return resized


def batch_to_spectrograms(iq_batch, verbose=True):
    """
    Convert a batch of IQ signals to spectrograms.
    
    Parameters:
        iq_batch: 2D array of shape (num_signals, num_samples).
        verbose: Print progress updates.
    
    Returns:
        4D numpy array of shape (num_signals, height, width, 1) — CNN-ready.
    """
    num_signals = len(iq_batch)
    h, w = cfg.SPEC_SIZE
    spectrograms = np.zeros((num_signals, h, w, 1), dtype=np.float32)
    
    for i in range(num_signals):
        spec = iq_to_spectrogram(iq_batch[i])
        spectrograms[i, :, :, 0] = spec.astype(np.float32)
        
        if verbose and (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{num_signals} spectrograms...")
    
    if verbose:
        print(f"  Spectrogram conversion complete: {spectrograms.shape}")
    
    return spectrograms


def save_spectrograms(spectrograms, labels, snrs, filepath=None):
    """Save processed spectrograms to disk."""
    import os
    filepath = filepath or os.path.join(cfg.DATA_DIR, "spectrograms.npz")
    np.savez_compressed(filepath,
                        spectrograms=spectrograms,
                        labels=labels,
                        snrs=snrs)
    print(f"Spectrograms saved to {filepath}")


def load_spectrograms(filepath=None):
    """Load previously saved spectrograms."""
    import os
    filepath = filepath or os.path.join(cfg.DATA_DIR, "spectrograms.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Spectrograms not found at {filepath}.")
    data = np.load(filepath)
    return data["spectrograms"], data["labels"], data["snrs"]


if __name__ == "__main__":
    from signal_generator import load_dataset
    
    print("Loading IQ dataset...")
    iq_data, labels, snrs = load_dataset()
    
    print("Converting to spectrograms...")
    specs = batch_to_spectrograms(iq_data)
    
    save_spectrograms(specs, labels, snrs)
    print(f"Output shape: {specs.shape}")
    print(f"Value range: [{specs.min():.4f}, {specs.max():.4f}]")
