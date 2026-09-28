"""
evaluate.py - Performance Evaluation: ROC curves, confusion matrices, accuracy plots.
Uses PyTorch for model inference.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import config as cfg
from cnn_model import predict_with_occupancy
from signal_generator import GENERATORS, _add_awgn
from spectrogram_processor import iq_to_spectrogram
from fusion_center import FusionCenter


def plot_training_history(history, save_dir=None):
    """Plot training/validation loss and accuracy curves."""
    save_dir = save_dir or cfg.RESULTS_DIR
    fig, axes = plt.subplots(1, 2, figsize=cfg.FIGSIZE_LARGE)
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['accuracy'], label='Train Acc')
    axes[1].plot(history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle('Training History', fontsize=16); plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_accuracy_vs_snr(model, X_test, y_test, snrs_test, save_dir=None):
    """Plot per-class and overall accuracy vs SNR."""
    save_dir = save_dir or cfg.RESULTS_DIR
    preds = predict_with_occupancy(model, X_test)
    pred_classes = preds['class_indices']
    unique_snrs = sorted(np.unique(snrs_test))
    overall_acc = []
    per_class_acc = {c: [] for c in cfg.CLASSES}
    for snr in unique_snrs:
        mask = snrs_test == snr
        if mask.sum() == 0: continue
        overall_acc.append((pred_classes[mask] == y_test[mask]).mean())
        for ci, cn in enumerate(cfg.CLASSES):
            cm = mask & (y_test == ci)
            per_class_acc[cn].append((pred_classes[cm] == ci).mean() if cm.sum() > 0 else np.nan)

    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_STANDARD)
    ax.plot(unique_snrs, overall_acc, 'k-o', lw=2.5, ms=6, label='Overall')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for ci, cn in enumerate(cfg.CLASSES):
        ax.plot(unique_snrs, per_class_acc[cn], '--', color=colors[ci], lw=1.5, marker='s', ms=4, label=cn)
    ax.set_xlabel('SNR (dB)', fontsize=13); ax.set_ylabel('Classification Accuracy', fontsize=13)
    ax.set_title('Classification Accuracy vs SNR', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])
    path = os.path.join(save_dir, "accuracy_vs_snr.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrices(model, X_test, y_test, snrs_test, save_dir=None, snr_values=None):
    """Plot confusion matrices at selected SNR values."""
    save_dir = save_dir or cfg.RESULTS_DIR
    snr_values = snr_values or [-10, 0, 10]
    preds = predict_with_occupancy(model, X_test)
    pred_classes = preds['class_indices']
    fig, axes = plt.subplots(1, len(snr_values), figsize=(6*len(snr_values), 5))
    if len(snr_values) == 1: axes = [axes]
    for idx, snr in enumerate(snr_values):
        mask = snrs_test == snr
        if mask.sum() == 0: continue
        cm = confusion_matrix(y_test[mask], pred_classes[mask], labels=list(range(cfg.NUM_CLASSES)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES, ax=axes[idx], vmin=0, vmax=1)
        axes[idx].set_title(f'SNR = {snr} dB', fontsize=13)
        axes[idx].set_xlabel('Predicted'); axes[idx].set_ylabel('True')
    plt.suptitle('Confusion Matrices', fontsize=15); plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_roc_individual(model, X_test, y_test, save_dir=None):
    """Plot per-class ROC curves for individual node sensing."""
    save_dir = save_dir or cfg.RESULTS_DIR
    preds = predict_with_occupancy(model, X_test)
    probs = preds['probabilities']
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_STANDARD)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for ci, cn in enumerate(cfg.CLASSES):
        y_bin = (y_test == ci).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, ci])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[ci], lw=2, label=f'{cn} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    ax.set_xlabel('Probability of False Alarm (Pf)', fontsize=13)
    ax.set_ylabel('Probability of Detection (Pd)', fontsize=13)
    ax.set_title('ROC Curves — Individual Node Sensing', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, "roc_individual.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_roc_cooperative(model, snr_values=None, num_trials=50, save_dir=None):
    """Plot ROC comparing Individual vs OR vs Majority Voting at selected SNRs."""
    save_dir = save_dir or cfg.RESULTS_DIR
    snr_values = snr_values or [-10, -5, 0, 5]
    threshold_range = np.linspace(0.1, 0.99, 25)
    fc = FusionCenter(model)
    fig, axes = plt.subplots(1, len(snr_values), figsize=(6*len(snr_values), 5))
    if len(snr_values) == 1: axes = [axes]

    for si, snr_db in enumerate(snr_values):
        ind_pf, ind_pd, or_pf, or_pd, maj_pf, maj_pd = [], [], [], [], [], []
        for thresh in threshold_range:
            id_c, if_c, od_c, of_c, md_c, mf_c = 0,0,0,0,0,0
            n_h1, n_h0 = 0, 0
            for _ in range(num_trials):
                cls = cfg.CLASSES[np.random.randint(len(cfg.CLASSES))]
                sig = GENERATORS[cls]()
                result = fc.cooperative_sense(sig, snr_db, thresh)
                for d in result['individual_decisions']:
                    if d['occupied']: id_c += 1
                n_h1 += len(result['individual_decisions'])
                if result['or_fusion']['global_occupied']: od_c += 1
                if result['majority_fusion']['global_occupied']: md_c += 1

                noise_decs = []
                for node in fc.nodes:
                    noise = _add_awgn(np.zeros(cfg.NUM_SAMPLES, dtype=complex), 100)
                    ns = iq_to_spectrogram(noise)
                    ni = ns.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
                    nr = predict_with_occupancy(model, ni, thresh)
                    noise_decs.append({'occupied': bool(nr['occupied'][0]),
                                       'confidence': float(nr['confidences'][0])})
                    if nr['occupied'][0]: if_c += 1
                n_h0 += len(fc.nodes)
                if FusionCenter.fuse_or(noise_decs)['global_occupied']: of_c += 1
                if FusionCenter.fuse_majority(noise_decs)['global_occupied']: mf_c += 1

            ind_pd.append(id_c/max(n_h1,1)); ind_pf.append(if_c/max(n_h0,1))
            or_pd.append(od_c/max(num_trials,1)); or_pf.append(of_c/max(num_trials,1))
            maj_pd.append(md_c/max(num_trials,1)); maj_pf.append(mf_c/max(num_trials,1))

        axes[si].plot(ind_pf, ind_pd, 'b-o', ms=3, lw=1.5, label='Individual')
        axes[si].plot(or_pf, or_pd, 'r-s', ms=3, lw=1.5, label='OR Rule')
        axes[si].plot(maj_pf, maj_pd, 'g-^', ms=3, lw=1.5, label='Majority')
        axes[si].plot([0,1],[0,1],'k--',alpha=0.3)
        axes[si].set_title(f'SNR = {snr_db} dB', fontsize=13)
        axes[si].set_xlabel('Pf'); axes[si].set_ylabel('Pd')
        axes[si].legend(fontsize=9); axes[si].grid(True, alpha=0.3)
        print(f"  ROC for SNR={snr_db} dB complete")

    plt.suptitle('ROC: Individual vs Cooperative Sensing', fontsize=15); plt.tight_layout()
    path = os.path.join(save_dir, "roc_cooperative.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_sample_spectrograms(X_test, y_test, snrs_test, save_dir=None):
    """Plot sample spectrograms for each class at different SNRs."""
    save_dir = save_dir or cfg.RESULTS_DIR
    snr_samples = [-10, 0, 10]
    fig, axes = plt.subplots(len(cfg.CLASSES), len(snr_samples),
                             figsize=(4*len(snr_samples), 4*len(cfg.CLASSES)))
    for ci, cn in enumerate(cfg.CLASSES):
        for si, snr in enumerate(snr_samples):
            mask = (y_test == ci) & (snrs_test == snr)
            idxs = np.where(mask)[0]
            if len(idxs) == 0: continue
            img = X_test[idxs[0], :, :, 0]
            axes[ci, si].imshow(img, aspect='auto', origin='lower', cmap='viridis')
            axes[ci, si].set_title(f'{cn} @ {snr} dB', fontsize=11)
    plt.suptitle('Sample Spectrograms', fontsize=15); plt.tight_layout()
    path = os.path.join(save_dir, "sample_spectrograms.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def generate_metrics_summary(model, X_test, y_test, snrs_test, save_dir=None):
    """Generate and save text summary of all metrics."""
    save_dir = save_dir or cfg.RESULTS_DIR
    preds = predict_with_occupancy(model, X_test)
    report = classification_report(y_test, preds['class_indices'], target_names=cfg.CLASSES)
    lines = ["="*60, "COOPERATIVE SPECTRUM SENSING — METRICS SUMMARY", "="*60, "",
             "Classification Report (All SNRs):", report, ""]
    lines.append("Accuracy by SNR:")
    for snr in sorted(np.unique(snrs_test)):
        mask = snrs_test == snr
        acc = (preds['class_indices'][mask] == y_test[mask]).mean()
        lines.append(f"  SNR={snr:+6.1f} dB: {acc:.4f}")
    lines += ["", f"Channel Occupied rate: {preds['occupied'].mean():.4f}",
              f"Mean confidence: {preds['confidences'].mean():.4f}", "="*60]
    summary = "\n".join(lines)
    path = os.path.join(save_dir, "metrics_summary.txt")
    with open(path, 'w') as f: f.write(summary)
    print(summary)
    print(f"\nSaved: {path}")


def run_full_evaluation(model, history, X_test, y_test, snrs_test,
                        cooperative_snrs=None, num_coop_trials=30):
    """Run all evaluation steps."""
    print("\n" + "="*60 + "\nRUNNING FULL EVALUATION\n" + "="*60)
    if history is not None:
        print("\n[1/6] Training history..."); plot_training_history(history)
    print("\n[2/6] Accuracy vs SNR..."); plot_accuracy_vs_snr(model, X_test, y_test, snrs_test)
    print("\n[3/6] Confusion matrices..."); plot_confusion_matrices(model, X_test, y_test, snrs_test)
    print("\n[4/6] Individual ROC curves..."); plot_roc_individual(model, X_test, y_test)
    print("\n[5/6] Sample spectrograms..."); plot_sample_spectrograms(X_test, y_test, snrs_test)
    print("\n[6/6] Cooperative ROC (may take a while)...")
    plot_roc_cooperative(model, snr_values=cooperative_snrs or [-10,-5,0,5], num_trials=num_coop_trials)
    print("\nMetrics summary..."); generate_metrics_summary(model, X_test, y_test, snrs_test)
    print("\n" + "="*60 + "\nEVALUATION COMPLETE\n" + "="*60)
