"""
main.py - End-to-end orchestrator for Cooperative Spectrum Sensing Simulation.
"""

import argparse
import os
import sys
import numpy as np
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 1: Cooperative Spectrum Sensing')
    parser.add_argument('--skip-training', action='store_true', help='Load saved model')
    parser.add_argument('--regenerate', action='store_true', help='Force dataset regeneration')
    parser.add_argument('--epochs', type=int, default=None, help=f'Max epochs (default: {cfg.EPOCHS})')
    parser.add_argument('--coop-trials', type=int, default=30, help='MC trials for cooperative ROC')
    parser.add_argument('--coop-snrs', nargs='+', type=int, default=[-10,-5,0,5])
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from train import run_training_pipeline
    from evaluate import run_full_evaluation
    from cnn_model import build_cnn

    print("="*60)
    print("COOPERATIVE SPECTRUM SENSING SIMULATION — PHASE 1")
    print("="*60)
    print(f"Classes:       {cfg.CLASSES}")
    print(f"SNR Range:     {cfg.SNR_RANGE_DB[0]} to {cfg.SNR_RANGE_DB[-1]} dB")
    print(f"Samples/class: {cfg.SAMPLES_PER_CLASS_PER_SNR} per SNR")
    print(f"Nodes:         {cfg.NUM_NODES}")
    print(f"Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)

    if args.skip_training:
        print("\n[SKIP] Loading pre-trained model...")
        model_path = os.path.join(cfg.RESULTS_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} not found. Run without --skip-training first.")
            sys.exit(1)
        model = build_cnn()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        history = None
        test_path = os.path.join(cfg.DATA_DIR, "test_data.npz")
        if not os.path.exists(test_path):
            print(f"ERROR: {test_path} not found."); sys.exit(1)
        data = np.load(test_path)
        X_test, y_test, snrs_test = data['X_test'], data['y_test'], data['snrs_test']
    else:
        print("\n[STEP 1/2] TRAINING PIPELINE")
        model, history, (X_test, y_test, snrs_test) = \
            run_training_pipeline(regenerate=args.regenerate, epochs=args.epochs)

    print("\n[STEP 2/2] EVALUATION PIPELINE")
    run_full_evaluation(model, history, X_test, y_test, snrs_test,
                        cooperative_snrs=args.coop_snrs, num_coop_trials=args.coop_trials)

    print("\nResults in:", cfg.RESULTS_DIR)
    for f in sorted(os.listdir(cfg.RESULTS_DIR)):
        sz = os.path.getsize(os.path.join(cfg.RESULTS_DIR, f))
        print(f"  {f:40s} ({sz:,} bytes)")


if __name__ == "__main__":
    main()
