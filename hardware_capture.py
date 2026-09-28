"""
hardware_capture.py — RTL-SDR Hardware Integration.
Captures REAL signals from antenna and classifies them using the trained CNN.

USAGE:
    1. First train the model:   python main.py --epochs 20
    2. Then run this script:    python hardware_capture.py

REQUIRES:
    - RTL-SDR stick plugged in
    - Zadig WinUSB driver installed
    - Trained model at results/best_model.pth
"""

import os
import sys
import numpy as np
import torch

# Ensure local DLLs (rtlsdr.dll, libusb-1.0.dll, etc.) are in PATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in os.environ.get('PATH', ''):
    os.environ['PATH'] = CURRENT_DIR + ';' + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory') and os.path.exists(CURRENT_DIR):
    try:
        os.add_dll_directory(CURRENT_DIR)
    except Exception:
        pass

# Project imports
import config as cfg
from cnn_model import build_cnn, predict_with_occupancy
from spectrogram_processor import iq_to_spectrogram

# Try importing RTL-SDR
try:
    from rtlsdr import RtlSdr
    HARDWARE_AVAILABLE = True
except Exception as e:
    print(f"[DEBUG] pyrtlsdr import failed: {e}")
    HARDWARE_AVAILABLE = False


def capture_real_signal(center_freq=91.5e6, sample_rate=2.048e6,
                        gain='auto', num_samples=1024):
    """
    Capture REAL IQ samples from RTL-SDR antenna.

    Parameters:
        center_freq: Frequency to tune to in Hz (default: 91.5 MHz FM)
        sample_rate: Sampling rate in Hz
        gain: 'auto' or numeric dB value
        num_samples: Number of IQ samples to capture

    Returns:
        Complex IQ samples array
    """
    if not HARDWARE_AVAILABLE:
        print("[ERROR] pyrtlsdr not installed or RTL-SDR not connected!")
        print("        Install: pip install pyrtlsdr")
        print("        Make sure RTL-SDR stick is plugged in + Zadig driver installed")
        sys.exit(1)

    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = gain

        print(f"[SDR] Tuned to:    {center_freq/1e6:.3f} MHz")
        print(f"[SDR] Sample rate: {sample_rate/1e6:.3f} MHz")
        print(f"[SDR] Capturing {num_samples} IQ samples...")

        iq_samples = sdr.read_samples(num_samples)
    finally:
        sdr.close()

    return iq_samples.astype(np.complex64)


def simulate_sdr_signal():
    """Fallback: generate a fake signal when no hardware is available."""
    from signal_generator import generate_fm, _add_awgn
    print("[SIM] No RTL-SDR found — using SIMULATED FM signal")
    clean = generate_fm()
    noisy = _add_awgn(clean, snr_db=5.0)
    return noisy


def load_trained_model():
    """Load the trained CNN model from results/."""
    model_path = os.path.join(cfg.RESULTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at: {model_path}")
        print("        Run first: python main.py --epochs 20")
        sys.exit(1)

    model = build_cnn()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"[OK] Loaded trained model from {model_path}")
    return model


def classify_signal(model, iq_samples):
    """
    Classify a single IQ signal capture.

    Pipeline: IQ -> Spectrogram -> CNN -> Prediction
    """
    # Step 1: Convert IQ to spectrogram
    spec = iq_to_spectrogram(iq_samples)
    spec_input = spec.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)

    # Step 2: CNN inference
    result = predict_with_occupancy(model, spec_input)

    return {
        'signal_type': result['class_names'][0],
        'confidence': float(result['confidences'][0]),
        'occupied': bool(result['occupied'][0]),
        'all_probabilities': {
            cls: float(result['probabilities'][0][i])
            for i, cls in enumerate(cfg.CLASSES)
        }
    }


def scan_frequency_bands(model):
    """
    Scan multiple frequency bands and classify each.
    Shows which bands are occupied vs vacant.
    """
    bands = [
        ("FM Radio 91.5",   91.5e6),
        ("FM Radio 93.5",   93.5e6),
        ("FM Radio 98.3",   98.3e6),
        ("FM Radio 104.0", 104.0e6),
        ("Airband 121.5",  121.5e6),
        ("ISM 433 MHz",    433.0e6),
    ]

    print("\n" + "=" * 60)
    print("SPECTRUM SCAN RESULTS")
    print("=" * 60)
    print(f"{'Band':<20} {'Status':<12} {'Signal':<8} {'Confidence':<12}")
    print("-" * 60)

    for name, freq in bands:
        try:
            iq = capture_real_signal(center_freq=freq, num_samples=cfg.NUM_SAMPLES)
            result = classify_signal(model, iq)

            status = "OCCUPIED" if result['occupied'] else "VACANT"
            print(f"  {name:<18} {status:<12} {result['signal_type']:<8} "
                  f"{result['confidence']:.1%}")
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")

    print("=" * 60)


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("COOPERATIVE SPECTRUM SENSING — LIVE HARDWARE MODE")
    print("=" * 60)

    # Load trained CNN
    model = load_trained_model()

    # Capture signal (real hardware or simulation)
    if HARDWARE_AVAILABLE:
        try:
            iq_samples = capture_real_signal(
                center_freq=91.5e6,       # Change this to your frequency
                num_samples=cfg.NUM_SAMPLES
            )
            print(f"[OK] Captured {len(iq_samples)} REAL samples from antenna")
        except Exception as e:
            print(f"[WARN] Hardware error: {e}")
            iq_samples = simulate_sdr_signal()
    else:
        iq_samples = simulate_sdr_signal()

    # Classify the captured signal
    result = classify_signal(model, iq_samples)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"  Signal Type:      {result['signal_type']}")
    print(f"  Confidence:       {result['confidence']:.1%}")
    print(f"  Channel Status:   {'OCCUPIED' if result['occupied'] else 'VACANT'}")
    print(f"\n  Probabilities:")
    for cls, prob in result['all_probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:6s}: {prob:.3f} {bar}")
    print("=" * 60)

    # Optional: Scan multiple bands (only with real hardware)
    if HARDWARE_AVAILABLE:
        print("\nScan multiple frequency bands? (y/n)")
        if input("> ").strip().lower() == 'y':
            scan_frequency_bands(model)
