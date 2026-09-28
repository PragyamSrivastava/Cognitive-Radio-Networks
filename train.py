"""
train.py - Training Pipeline for the CNN Signal Classifier (PyTorch).
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import config as cfg
from signal_generator import generate_dataset, load_dataset
from spectrogram_processor import batch_to_spectrograms, save_spectrograms, load_spectrograms
from cnn_model import build_cnn


def prepare_data(regenerate=False):
    """Prepare dataset: generate signals then convert to spectrograms."""
    spec_path = os.path.join(cfg.DATA_DIR, "spectrograms.npz")
    if not regenerate and os.path.exists(spec_path):
        print("Loading cached spectrograms...")
        return load_spectrograms(spec_path)

    data_path = os.path.join(cfg.DATA_DIR, "synthetic_dataset.npz")
    if not regenerate and os.path.exists(data_path):
        print("Loading cached IQ dataset...")
        iq_data, labels, snrs = load_dataset()
    else:
        print("Generating synthetic IQ dataset...")
        iq_data, labels, snrs = generate_dataset()

    print("Converting IQ signals to spectrograms...")
    spectrograms = batch_to_spectrograms(iq_data)
    save_spectrograms(spectrograms, labels, snrs, spec_path)
    return spectrograms, labels, snrs


def split_data(spectrograms, labels, snrs):
    """Split into train/val/test, stratified by class and SNR."""
    strat_key = labels * 1000 + (snrs + 100).astype(int)

    X_tv, X_test, y_tv, y_test, s_tv, s_test = train_test_split(
        spectrograms, labels, snrs,
        test_size=cfg.TEST_SPLIT, random_state=cfg.RANDOM_SEED, stratify=strat_key)

    val_ratio = cfg.VAL_SPLIT / (1 - cfg.TEST_SPLIT)
    strat_tv = y_tv * 1000 + (s_tv + 100).astype(int)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_tv, y_tv, s_tv,
        test_size=val_ratio, random_state=cfg.RANDOM_SEED, stratify=strat_tv)

    print(f"Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test


def _to_torch(X, y, device):
    """Convert numpy (N,H,W,1) to torch tensors (N,1,H,W)."""
    X_t = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return X_t, y_t


def train_model(X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
    """Build, compile, and train the CNN with PyTorch."""
    epochs = epochs or cfg.EPOCHS
    batch_size = batch_size or cfg.BATCH_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = build_cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.LR_REDUCE_PATIENCE,
        factor=cfg.LR_REDUCE_FACTOR, min_lr=1e-6)

    X_tr, y_tr = _to_torch(X_train, y_train, device)
    X_vl, y_vl = _to_torch(X_val, y_val, device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_vl)
            val_loss = criterion(val_out, y_vl).item()
            val_acc = (val_out.argmax(1) == y_vl).float().mean().item()

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{epochs} — "
              f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(cfg.RESULTS_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(
        torch.load(os.path.join(cfg.RESULTS_DIR, "best_model.pth"),
                    weights_only=True))
    print(f"Best val_accuracy: {best_val_acc:.4f}")
    return model, history


def run_training_pipeline(regenerate=False, epochs=None):
    """Full pipeline: data prep -> split -> train -> save."""
    spectrograms, labels, snrs = prepare_data(regenerate)
    X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test = \
        split_data(spectrograms, labels, snrs)
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=epochs)

    test_path = os.path.join(cfg.DATA_DIR, "test_data.npz")
    np.savez_compressed(test_path, X_test=X_test, y_test=y_test, snrs_test=s_test)
    print(f"Test data saved to {test_path}")
    return model, history, (X_test, y_test, s_test)


if __name__ == "__main__":
    run_training_pipeline()
