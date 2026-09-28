<div align="center">

# 📡 Analyzing Cooperative Spectrum Sensing Using Machine Learning for Cognitive Radio Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![RTL--SDR](https://img.shields.io/badge/Hardware-RTL--SDR%20(R820T2)-brightgreen.svg?logo=hackster&logoColor=white)](https://www.rtl-sdr.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An End-to-End AI-Driven Cognitive Radio Framework with 2D STFT Spectrograms, Deep CNN Modulation Classification, Multi-Node Cooperative Decision Fusion (OR & Majority Voting), and Live Over-The-Air RTL-SDR Hardware Validation.**

</div>

---

## 📌 Executive Summary

Dynamic Spectrum Access (DSA) in Cognitive Radio Networks (CRNs) mitigates artificial spectrum scarcity by allowing unlicensed Secondary Users (SUs) to opportunistically access idle licensed frequency bands (**White Spaces**) without interfering with Primary Users (PUs).

Traditional Energy Detectors fail under severe noise uncertainty ($\text{SNR} < -5\text{ dB}$) and shadowing (the **Hidden Node Problem**). This project solves these challenges by combining:
1. **Short-Time Fourier Transform (STFT)** to map 1D complex $\text{I/Q}$ baseband signals into normalized 2D Spectrograms ($64\times 64$).
2. A **Deep 4-Block Convolutional Neural Network (CNN)** in PyTorch (422K parameters) for automatic modulation recognition (**AM, FM, BPSK, QPSK**) and channel occupancy detection.
3. A **3-Node Cooperative Fusion Center** implementing **OR-Rule** (maximum PU protection) and **Majority Voting** (balanced spectrum access) to overcome fading.
4. **Physical RTL-SDR Hardware Integration** for real-time over-the-air spectrum sensing across the $88\text{--}108\text{ MHz}$ FM broadcast band with an interactive **Streamlit Web Dashboard**.

---

## 🏗️ System Architecture & Pipeline

```
                                SYSTEM PIPELINE FLOWCHART
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  [ RF Source ]  ──► (Synthetic Generator: AM, FM, BPSK, QPSK / Real RTL-SDR Antenna)  │
│        │                                                                               │
│        ▼                                                                               │
│  [ Baseband I/Q Buffer ] ──► Length N = 1024 complex samples: r[n] = I[n] + j·Q[n]     │
│        │                                                                               │
│        ▼                                                                               │
│  [ STFT Processor ] ──► Hann Window (N=64, 75% Overlap) ──► PSD Decibel Normalization │
│        │                                                                               │
│        ▼                                                                               │
│  [ 2D Spectrogram Matrix ] ──► 64 × 64 Grayscale Time-Frequency Image                  │
│        │                                                                               │
│        ▼                                                                               │
│  [ Deep 4-Block CNN ]                                                                  │
│    ├── Conv2D(32)  + BatchNorm + MaxPool(2x2)                                          │
│    ├── Conv2D(64)  + BatchNorm + MaxPool(2x2)                                          │
│    ├── Conv2D(128) + BatchNorm + MaxPool(2x2)                                          │
│    ├── Conv2D(256) + BatchNorm + MaxPool(2x2)                                          │
│    └── Classifier: GlobalAvgPool + Dense(128) + Dropout(0.5) + Softmax(4 Classes)      │
│        │                                                                               │
│        ▼                                                                               │
│  [ Distributed Sensing Nodes ] ──► Node 1, Node 2, Node 3 (Independent Fading/Noise)   │
│        │                                                                               │
│        ▼                                                                               │
│  [ Centralized Fusion Center ]                                                         │
│    ├── OR Rule (1-out-of-3): Qd = 1 - (1 - Pd)^3  (Maximizes PU Protection)           │
│    └── Majority Vote (2-out-of-3): Qd = 3·Pd²·(1-Pd) + Pd³  (Balances Detection/FA)    │
│        │                                                                               │
│        ▼                                                                               │
│  [ Dynamic Decision ] ──► OCCUPIED (Backoff)  /  VACANT (White Space: Transmit Data)   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Signal Hypotheses
$$\mathcal{H}_0 : r(t) = w(t) \quad \text{(Channel Vacant / Noise Only)}$$
$$\mathcal{H}_1 : r(t) = h(t) * s(t) + w(t) \quad \text{(Primary User Active)}$$

### 2. Time-Frequency Transformation (STFT)
$$\text{STFT}\{r[n]\}(m, \omega) = X(m, \omega) = \sum_{n=-\infty}^{\infty} r[n] \cdot w[n - mR] \cdot e^{-j \omega n}$$
$$P(m, k) = 10 \log_{10}\left( |X(m, k)|^2 + 10^{-12} \right)$$

### 3. Cooperative Decision Fusion
* **OR Rule ($M = 1$):** $Q_d = 1 - \prod_{i=1}^K (1 - P_{d,i})$, $Q_f = 1 - \prod_{i=1}^K (1 - P_{f,i})$
* **Majority Voting ($M = \lceil K/2 \rceil$):** $Q_d = \sum_{l=\lceil K/2 \rceil}^K \binom{K}{l} P_d^l (1 - P_d)^{K-l}$

---

## 📊 Experimental Results

| Metric | Result | Notes |
|---|---|---|
| **Peak Model Accuracy** | **95.45%** | At high SNR ($\ge +8\text{ dB}$) |
| **Accuracy at $-4\text{ dB}$ SNR** | **> 90.0%** | Exceeds classical energy detection |
| **Single-Node $P_d$ (at $-10\text{ dB}$)** | **0.68** | Degraded by noise & fading |
| **Cooperative OR-Rule $Q_d$ (at $-10\text{ dB}$)** | **0.94** | **$+26\%$ Cooperative Detection Gain** |
| **AM Classification F1-Score** | **0.95** | Continuous carrier ridge |
| **FM Classification F1-Score** | **0.95** | Frequency deviation band |
| **Total Model Parameters** | **422,212** | Ultra-lightweight & edge-deployable |

### Live Over-The-Air Hardware Scan (RTL-SDR R820T2)
```text
============================================================
SPECTRUM SCAN RESULTS (Bangalore Over-The-Air Capture)
============================================================
Center Freq   Observed Class   Confidence   Channel State   Cognitive Action
----------------------------------------------------------------------------
91.5 MHz      FM Broadcast     91.6%        OCCUPIED        Back-off (PU Active)
93.5 MHz      FM Broadcast     92.3%        OCCUPIED        Back-off (PU Active)
98.3 MHz      Noise Floor      49.1%        VACANT          ✨ SPECTRUM HOLE FOUND
104.0 MHz     FM Broadcast     89.3%        OCCUPIED        Back-off (PU Active)
121.5 MHz     Airband (AM)     88.4%        OCCUPIED        Emergency Band (Hold)
433.0 MHz     ISM (BPSK)       82.2%        OCCUPIED        IoT Telemetry (Hold)
============================================================
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/PragyamSrivastava/Child-Safety-Wearable-Device.git
cd Child-Safety-Wearable-Device
pip install -r requirements.txt
```

### 2. Launch Interactive Web Dashboard
```bash
streamlit run dashboard.py
```
👉 Open your browser at **`http://localhost:8501`** to interact with:
* **Tab 1:** Live Spectrum Scanner (Single-channel capture & AI prediction)
* **Tab 2:** Multi-Channel White-Space Map (Wideband sweep)
* **Tab 3:** Cooperative Fusion Center (3-Node simulation with OR vs Majority voting)
* **Tab 4:** Model Analytics & ROC Performance Curves
* **Tab 5:** Phase 2 Advanced Systems (Autonomous Hopping, Soft Fusion, SSDF Security Defense)

### 3. Run Simulation & Train CNN (CLI)
```bash
python main.py --epochs 10 --coop-trials 10
```

### 4. Capture Live RTL-SDR Signals (Terminal Mode)
```bash
python hardware_capture.py
```

---

## 📁 Repository Structure

```
├── config.py                 # Central configuration & tunable hyperparameters
├── signal_generator.py       # AM, FM, BPSK, QPSK IQ generation + AWGN channel
├── spectrogram_processor.py  # STFT signal processing & 2D spectrogram conversion
├── cnn_model.py              # PyTorch 4-block deep CNN architecture
├── fusion_center.py          # Cooperative 3-node sensing & fusion center logic
├── phase2_engine.py          # Autonomous hopping, soft fusion & SSDF security defense
├── train.py                  # Stratified dataset split & training pipeline
├── evaluate.py               # ROC curves, confusion matrices & accuracy evaluations
├── hardware_capture.py       # Live RTL-SDR USB hardware-in-the-loop capture
├── dashboard.py              # Real-time Streamlit web analytics dashboard
├── generate_presentation.py  # Automated PPTX presentation generator
├── generate_poster.py        # A3 academic poster generator
├── presentation.md           # 12-slide comprehensive presentation script
├── requirements.txt          # Python package dependencies
├── .gitignore                # Git ignore rules
└── results/                  # Generated plots & model weights
    ├── roc_cooperative.png
    ├── roc_individual.png
    ├── accuracy_vs_snr.png
    ├── confusion_matrices.png
    ├── sample_spectrograms.png
    ├── training_history.png
    ├── metrics_summary.txt
    └── best_model.pth
```

---

## 👥 Project Team & Credits

* **Institution:** Department of Electronics & Telecommunication Engineering, **Bangalore Institute of Technology**, Bengaluru, India
* **Project Guide:** Prof. Asha R
* **Team Members:**
  * **Pragyam Srivastava** (`1BI23ET041`)
  * **Sangeetha M** (`1BI23ET050`)
  * **Rachana M N** (`1BI23ET064`)
  * **Amoghapriya R B** (`1BI23ET066`)

---

## 📜 References

1. I. F. Akyildiz, W.-Y. Lee, M. C. Vuran, and S. Mohanty, *"A survey on spectrum management in cognitive radio networks,"* IEEE Communications Magazine, vol. 46, no. 4, pp. 40–48, 2008.
2. J. Mitola and G. Q. Maguire, *"Cognitive radio: Making software radios more personal,"* IEEE Personal Communications, vol. 6, no. 4, pp. 13–18, 1999.
3. T. J. O'Shea and J. Hoydis, *"An introduction to deep learning for the physical layer,"* IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 4, pp. 563–575, 2017.
4. Y. Sun, J. Wang, and Z. Han, *"Deep learning-based cooperative spectrum sensing in cognitive radio networks,"* IEEE Transactions on Wireless Communications, vol. 18, no. 6, pp. 3126–3139, 2019.
5. S. Rajendran, W. Meert, V. Lenders, and S. Pollin, *"Deep learning models for wireless signal classification with distributed low-cost spectrum sensors,"* IEEE TCCN, vol. 4, no. 3, pp. 433–445, 2018.

---
<div align="center">
Made with ❤️ by Team G10, Dept. of ETE, Bangalore Institute of Technology
</div>
