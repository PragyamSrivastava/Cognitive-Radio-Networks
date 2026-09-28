"""
signal_generator.py — Synthetic IQ Signal Generation.
Generates raw complex IQ samples for AM, FM, BPSK, and QPSK modulations
with configurable AWGN noise levels.
"""

import numpy as np
from scipy.signal import firwin, lfilter
import os
import config as cfg


def _time_vector(num_samples=None, sample_rate=None):
    """Generate time vector for signal generation."""
    n = num_samples or cfg.NUM_SAMPLES
    fs = sample_rate or cfg.SAMPLE_RATE
    return np.arange(n) / fs


def _add_awgn(signal, snr_db):
    """
    Add Additive White Gaussian Noise to a complex signal.
    
    Parameters:
        signal: Complex IQ signal array.
        snr_db: Target SNR in dB.
    
    Returns:
        Noisy complex signal.
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def _raised_cosine_filter(num_taps, rolloff, symbol_period, sample_rate):
    """
    Generate a raised cosine pulse-shaping filter.
    
    Parameters:
        num_taps: Number of filter taps.
        rolloff: Roll-off factor (0 to 1).
        symbol_period: Symbol period in seconds.
        sample_rate: Sampling rate in Hz.
    
    Returns:
        Normalized filter coefficients.
    """
    Ts = symbol_period
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sample_rate
    
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 / Ts
        elif rolloff != 0 and abs(abs(ti) - Ts / (2 * rolloff)) < 1e-12:
            h[i] = (rolloff / (2 * Ts)) * np.sin(np.pi / (2 * rolloff))
        else:
            numerator = np.sin(np.pi * ti / Ts) * np.cos(np.pi * rolloff * ti / Ts)
            denominator = (np.pi * ti / Ts) * (1 - (2 * rolloff * ti / Ts) ** 2)
            if abs(denominator) < 1e-12:
                h[i] = 1.0 / Ts
            else:
                h[i] = numerator / denominator / Ts
    
    # Normalize
    h /= np.sqrt(np.sum(h ** 2))
    return h


def generate_am(num_samples=None, sample_rate=None, carrier_freq=None,
                message_freq=None, mod_index=None):
    """
    Generate AM (DSB-FC) IQ signal.
    
    s(t) = (1 + m * msg(t)) * exp(j * 2π * fc * t)
    """
    t = _time_vector(num_samples, sample_rate)
    fc = carrier_freq or cfg.CARRIER_FREQ
    fm = message_freq or cfg.MESSAGE_FREQ
    m = mod_index or cfg.AM_MODULATION_INDEX
    
    message = np.sin(2 * np.pi * fm * t)
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    signal = (1 + m * message) * carrier
    
    # Normalize to unit power
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))
    return signal


def generate_fm(num_samples=None, sample_rate=None, carrier_freq=None,
                message_freq=None, freq_deviation=None):
    """
    Generate FM IQ signal.
    
    s(t) = exp(j * (2π * fc * t + 2π * Δf * ∫msg(τ)dτ))
    """
    t = _time_vector(num_samples, sample_rate)
    fc = carrier_freq or cfg.CARRIER_FREQ
    fm = message_freq or cfg.MESSAGE_FREQ
    delta_f = freq_deviation or cfg.FM_FREQ_DEVIATION
    fs = sample_rate or cfg.SAMPLE_RATE
    
    message = np.sin(2 * np.pi * fm * t)
    # Cumulative integral of message signal
    integral = np.cumsum(message) / fs
    phase = 2 * np.pi * fc * t + 2 * np.pi * delta_f * integral
    signal = np.exp(1j * phase)
    
    # Normalize to unit power
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))
    return signal


def generate_bpsk(num_samples=None, sample_rate=None, carrier_freq=None,
                  symbol_rate=None, rolloff=None):
    """
    Generate BPSK IQ signal with raised cosine pulse shaping.
    
    Random bits → NRZ → pulse shaping → carrier modulation.
    """
    n = num_samples or cfg.NUM_SAMPLES
    fs = sample_rate or cfg.SAMPLE_RATE
    fc = carrier_freq or cfg.CARRIER_FREQ
    sym_rate = symbol_rate or cfg.SYMBOL_RATE
    beta = rolloff or cfg.ROLLOFF
    
    samples_per_symbol = int(fs / sym_rate)
    num_symbols = int(np.ceil(n / samples_per_symbol)) + 10  # Extra for filter
    
    # Random bits → NRZ mapping: 0 → -1, 1 → +1
    bits = np.random.randint(0, 2, num_symbols)
    nrz = 2 * bits - 1  # {-1, +1}
    
    # Upsample
    upsampled = np.zeros(num_symbols * samples_per_symbol)
    upsampled[::samples_per_symbol] = nrz
    
    # Pulse shaping with raised cosine
    num_taps = 6 * samples_per_symbol + 1
    rc_filter = _raised_cosine_filter(num_taps, beta, 1.0 / sym_rate, fs)
    shaped = np.convolve(upsampled, rc_filter, mode='same')
    
    # Truncate to desired length
    shaped = shaped[:n]
    
    # Carrier modulation
    t = _time_vector(n, fs)
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    signal = shaped * carrier
    
    # Normalize to unit power
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))
    return signal


def generate_qpsk(num_samples=None, sample_rate=None, carrier_freq=None,
                  symbol_rate=None, rolloff=None):
    """
    Generate QPSK IQ signal with raised cosine pulse shaping.
    
    Random symbol pairs → I/Q mapping → pulse shaping → carrier modulation.
    """
    n = num_samples or cfg.NUM_SAMPLES
    fs = sample_rate or cfg.SAMPLE_RATE
    fc = carrier_freq or cfg.CARRIER_FREQ
    sym_rate = symbol_rate or cfg.SYMBOL_RATE
    beta = rolloff or cfg.ROLLOFF
    
    samples_per_symbol = int(fs / sym_rate)
    num_symbols = int(np.ceil(n / samples_per_symbol)) + 10
    
    # Random symbol pairs → QPSK constellation: {±1 ± j} / sqrt(2)
    bits_i = np.random.randint(0, 2, num_symbols)
    bits_q = np.random.randint(0, 2, num_symbols)
    symbols_i = (2 * bits_i - 1) / np.sqrt(2)
    symbols_q = (2 * bits_q - 1) / np.sqrt(2)
    
    # Upsample I and Q
    upsampled_i = np.zeros(num_symbols * samples_per_symbol)
    upsampled_q = np.zeros(num_symbols * samples_per_symbol)
    upsampled_i[::samples_per_symbol] = symbols_i
    upsampled_q[::samples_per_symbol] = symbols_q
    
    # Pulse shaping
    num_taps = 6 * samples_per_symbol + 1
    rc_filter = _raised_cosine_filter(num_taps, beta, 1.0 / sym_rate, fs)
    shaped_i = np.convolve(upsampled_i, rc_filter, mode='same')[:n]
    shaped_q = np.convolve(upsampled_q, rc_filter, mode='same')[:n]
    
    # Combine I/Q and apply carrier
    baseband = shaped_i + 1j * shaped_q
    t = _time_vector(n, fs)
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    signal = baseband * carrier
    
    # Normalize to unit power
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))
    return signal


# Map class names to generator functions
GENERATORS = {
    "AM": generate_am,
    "FM": generate_fm,
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
}


def generate_dataset(snr_range=None, samples_per_class_per_snr=None, seed=None):
    """
    Generate the full synthetic dataset.
    
    Returns:
        signals: List of dicts with keys {signal_type, snr_db, iq_data}.
        Also saves to disk as .npz for reproducibility.
    """
    snr_range = snr_range if snr_range is not None else cfg.SNR_RANGE_DB
    n_samples = samples_per_class_per_snr or cfg.SAMPLES_PER_CLASS_PER_SNR
    rng_seed = seed if seed is not None else cfg.RANDOM_SEED
    np.random.seed(rng_seed)
    
    all_iq = []
    all_labels = []
    all_snrs = []
    
    total = len(cfg.CLASSES) * len(snr_range) * n_samples
    count = 0
    
    for class_idx, class_name in enumerate(cfg.CLASSES):
        gen_func = GENERATORS[class_name]
        for snr_db in snr_range:
            for _ in range(n_samples):
                clean_signal = gen_func()
                noisy_signal = _add_awgn(clean_signal, snr_db)
                
                all_iq.append(noisy_signal)
                all_labels.append(class_idx)
                all_snrs.append(snr_db)
                
                count += 1
                if count % 2000 == 0:
                    print(f"  Generated {count}/{total} signals...")
    
    all_iq = np.array(all_iq)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)
    
    # Save to disk
    save_path = os.path.join(cfg.DATA_DIR, "synthetic_dataset.npz")
    np.savez_compressed(save_path,
                        iq_data=all_iq,
                        labels=all_labels,
                        snrs=all_snrs)
    print(f"Dataset saved to {save_path}")
    print(f"  Shape: {all_iq.shape}, Labels: {all_labels.shape}, SNRs: {all_snrs.shape}")
    
    return all_iq, all_labels, all_snrs


def load_dataset():
    """Load previously generated dataset from disk."""
    path = os.path.join(cfg.DATA_DIR, "synthetic_dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run generate_dataset() first.")
    data = np.load(path)
    return data["iq_data"], data["labels"], data["snrs"]


if __name__ == "__main__":
    print("Generating synthetic IQ dataset...")
    iq, labels, snrs = generate_dataset()
    print(f"Done. Total samples: {len(labels)}")
    for i, cls in enumerate(cfg.CLASSES):
        print(f"  {cls}: {np.sum(labels == i)} samples")
