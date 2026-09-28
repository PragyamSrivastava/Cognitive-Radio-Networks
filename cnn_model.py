"""
cnn_model.py - CNN Architecture for Multi-Class Signal Classification (PyTorch).
Lightweight CNN for 128x128 single-channel spectrogram images.
"""

import numpy as np
import torch
import torch.nn as nn
import config as cfg


class SignalCNN(nn.Module):
    """
    4-block CNN for spectrogram classification.
    
    Architecture:
        Conv2D(32) -> BatchNorm -> ReLU -> MaxPool
        Conv2D(64) -> BatchNorm -> ReLU -> MaxPool
        Conv2D(128) -> BatchNorm -> ReLU -> MaxPool
        Conv2D(256) -> BatchNorm -> ReLU -> MaxPool
        GlobalAvgPool -> Dense(128) -> Dropout -> Dense(num_classes)
    """
    def __init__(self, num_classes=None, dropout_rate=None):
        super().__init__()
        num_classes = num_classes or cfg.NUM_CLASSES
        dropout_rate = dropout_rate or cfg.DROPOUT_RATE

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_cnn(num_classes=None, dropout_rate=None):
    """Build and return the CNN model."""
    model = SignalCNN(num_classes, dropout_rate)
    total = sum(p.numel() for p in model.parameters())
    print(f"CNN built — {total:,} parameters")
    return model


def predict_with_occupancy(model, spectrograms, threshold=None, device=None):
    """
    Run inference: predict signal type + channel occupancy.

    Parameters:
        model: Trained SignalCNN.
        spectrograms: numpy array (N, H, W, 1) or (N, 1, H, W).
        threshold: Confidence threshold for occupied.
        device: torch device.

    Returns:
        Dict with class_indices, class_names, confidences, occupied, probabilities.
    """
    threshold = threshold or cfg.OCCUPIED_THRESHOLD
    device = device or torch.device('cpu')
    model.eval()

    # Convert (N,H,W,1) -> (N,1,H,W) if needed
    if isinstance(spectrograms, np.ndarray):
        if spectrograms.ndim == 4 and spectrograms.shape[-1] == 1:
            spectrograms = spectrograms.transpose(0, 3, 1, 2)
        x = torch.tensor(spectrograms, dtype=torch.float32, device=device)
    else:
        x = spectrograms.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    class_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    return {
        'class_indices': class_indices,
        'class_names': np.array([cfg.CLASSES[i] for i in class_indices]),
        'confidences': confidences,
        'occupied': confidences >= threshold,
        'probabilities': probs,
    }


if __name__ == "__main__":
    model = build_cnn()
    print(model)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print(f"Output shape: {out.shape}")
