# 📡 Analyzing Cooperative Spectrum Sensing Using Machine Learning for Cognitive Radio Networks

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![RTL-SDR](https://img.shields.io/badge/Hardware-RTL--SDR%20\(R820T2\)-brightgreen.svg?logo=hackster\&logoColor=white)](https://www.rtl-sdr.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An End-to-End AI-Driven Cognitive Radio Framework with 2D STFT Spectrograms, Deep CNN Modulation Classification, Multi-Node Cooperative Decision Fusion, and RTL-SDR Hardware Validation.**

</div>

---

## 📌 Executive Summary

Dynamic Spectrum Access (DSA) in Cognitive Radio Networks (CRNs) helps address spectrum scarcity by allowing unlicensed Secondary Users (SUs) to opportunistically access idle licensed frequency bands (**White Spaces**) without interfering with Primary Users (PUs).

Traditional Energy Detectors can experience degraded performance under severe noise uncertainty and fading conditions. This project addresses these challenges using a machine-learning-based cooperative spectrum sensing framework that combines:

1. **Short-Time Fourier Transform (STFT)** to convert 1D complex I/Q baseband signals into normalized 2D spectrograms.
2. A **Deep 4-Block Convolutional Neural Network (CNN)** implemented using PyTorch for automatic modulation recognition and channel occupancy detection.
3. A **3-Node Cooperative Fusion Center** implementing **OR Rule** and **Majority Voting** for cooperative spectrum sensing.
4. **RTL-SDR Hardware Integration** for real-time over-the-air spectrum sensing.
5. An interactive **Streamlit Web Dashboard** for visualization, prediction, spectrum scanning, and cooperative sensing analysis.

---

## 🏗️ System Architecture & Pipeline

```text
                         SYSTEM PIPELINE FLOWCHART

┌──────────────────────────────────────────────────────────────────────────────┐
│ [ RF Source ]                                                               │
│      │                                                                       │
│      ├── Synthetic Generator: AM, FM, BPSK, QPSK                             │
│      └── Real RTL-SDR Antenna                                                │
│      │                                                                       │
│      ▼                                                                       │
│ [ Baseband I/Q Buffer ]                                                      │
│      │                                                                       │
│      │  N = 1024 complex samples: r[n] = I[n] + jQ[n]                       │
│      ▼                                                                       │
│ [ STFT Processor ]                                                           │
│      │                                                                       │
│      │  Hann Window (N=64, 75% Overlap)                                     │
│      │  PSD Decibel Normalization                                            │
│      ▼                                                                       │
│ [ 2D Spectrogram Matrix ]                                                    │
│      │                                                                       │
│      │  64 × 64 Grayscale Time-Frequency Image                               │
│      ▼                                                                       │
│ [ Deep 4-Block CNN ]                                                         │
│      │                                                                       │
│      ├── Conv2D(32)  + BatchNorm + MaxPool(2×2)                             │
│      ├── Conv2D(64)  + BatchNorm + MaxPool(2×2)                             │
│      ├── Conv2D(128) + BatchNorm + MaxPool(2×2)                             │
│      ├── Conv2D(256) + BatchNorm + MaxPool(2×2)                             │
│      └── GlobalAvgPool + Dense(128) + Dropout(0.5) + Softmax                 │
│      │                                                                       │
│      ▼                                                                       │
│ [ Distributed Sensing Nodes ]                                                │
│      │                                                                       │
│      └── Node 1, Node 2, Node 3                                             │
│          Independent Noise/Fading Conditions                                  │
│      │                                                                       │
│      ▼                                                                       │
│ [ Centralized Fusion Center ]                                                │
│      │                                                                       │
│      ├── OR Rule       → Maximum PU Protection                               │
│      └── Majority Vote → Balanced Detection / False Alarm                   │
│      │                                                                       │
│      ▼                                                                       │
│ [ Dynamic Decision ]                                                         │
│      │                                                                       │
│      ├── OCCUPIED → Backoff                                                 │
│      └── VACANT    → Spectrum Hole / Transmit Data                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Signal Hypotheses

Under the two-state spectrum sensing model:

$$
\mathcal{H}_0 : r(t) = w(t)
$$

**Channel Vacant / Noise Only**

$$
\mathcal{H}_1 : r(t) = h(t) * s(t) + w(t)
$$

**Primary User Active**

Where:

* $r(t)$ = received signal
* $s(t)$ = primary user signal
* $h(t)$ = wireless channel response
* $w(t)$ = additive noise

### 2. Time-Frequency Transformation — STFT

$$
\text{STFT}\{r[n]\}(m,\omega)
=
X(m,\omega)
=
\sum_{n=-\infty}^{\infty}
r[n]w[n-mR]e^{-j\omega n}
$$

Power spectral density is represented as:

$$
P(m,k)
=
10\log_{10}
\left(
|X(m,k)|^2 + 10^{-12}
\right)
$$

### 3. Cooperative Decision Fusion

**OR Rule:**

$$
Q_d
=
1-\prod_{i=1}^{K}(1-P_{d,i})
$$

$$
Q_f
=
1-\prod_{i=1}^{K}(1-P_{f,i})
$$

**Majority Voting:**

$$
Q_d
=
\sum_{l=\lceil K/2\rceil}^{K}
\binom{K}{l}
P_d^l
(1-P_d)^{K-l}
$$

---

## 🧠 Deep Learning Model

The project uses a **4-block CNN architecture** implemented in PyTorch.

### CNN Architecture

| Layer           | Configuration                     |
| --------------- | --------------------------------- |
| Input           | 64 × 64 Spectrogram               |
| Conv Block 1    | 32 Filters + BatchNorm + MaxPool  |
| Conv Block 2    | 64 Filters + BatchNorm + MaxPool  |
| Conv Block 3    | 128 Filters + BatchNorm + MaxPool |
| Conv Block 4    | 256 Filters + BatchNorm + MaxPool |
| Global Pooling  | Global Average Pooling            |
| Fully Connected | Dense Layer — 128 Units           |
| Regularization  | Dropout — 0.5                     |
| Output          | 4 Classes                         |

### Supported Modulation Classes

* AM — Amplitude Modulation
* FM — Frequency Modulation
* BPSK — Binary Phase Shift Keying
* QPSK — Quadrature Phase Shift Keying

---

## 🤝 Cooperative Spectrum Sensing

The system uses multiple sensing nodes to improve spectrum detection under fading and noise.

### Node Configuration

```text
              ┌───────────────┐
              │    Node 1     │
              └───────┬───────┘
                      │
                      │
              ┌───────▼───────┐
              │    Node 2     │
              └───────┬───────┘
                      │
                      │
              ┌───────▼───────┐
              │    Node 3     │
              └───────┬───────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Fusion Center  │
             └────────┬────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
          OR Rule        Majority Vote
             │                 │
             └────────┬────────┘
                      ▼
              Final Decision
```

### OR Rule

The spectrum is considered occupied if **at least one sensing node** detects the presence of a Primary User.

This provides stronger protection against missed detections.

### Majority Voting

The spectrum is considered occupied when **at least two out of three sensing nodes** report an occupied channel.

This provides a balance between detection performance and false alarms.

---

## 📊 Experimental Results

| Metric                                  |      Result | Notes                        |
| --------------------------------------- | ----------: | ---------------------------- |
| **Peak Model Accuracy**                 |  **95.45%** | At high SNR (≥ +8 dB)        |
| **Accuracy at -4 dB SNR**               | **> 90.0%** | CNN-based classification     |
| **Single-Node $P_d$ at -10 dB**         |    **0.68** | Affected by noise and fading |
| **Cooperative OR-Rule $Q_d$ at -10 dB** |    **0.94** | Cooperative detection gain   |
| **AM Classification F1-Score**          |    **0.95** | Continuous carrier ridge     |
| **FM Classification F1-Score**          |    **0.95** | Frequency deviation band     |
| **Total Model Parameters**              | **422,212** | Lightweight CNN architecture |

---

## 📡 Live Over-The-Air Hardware Scan

The project supports live spectrum sensing using an **RTL-SDR R820T2** receiver.

Example spectrum scan:

```text
============================================================
SPECTRUM SCAN RESULTS
============================================================

Center Freq   Observed Class   Confidence   Channel State
------------------------------------------------------------
91.5 MHz      FM Broadcast     91.6%        OCCUPIED
93.5 MHz      FM Broadcast     92.3%        OCCUPIED
98.3 MHz      Noise Floor      49.1%        VACANT
104.0 MHz     FM Broadcast     89.3%        OCCUPIED
121.5 MHz     Airband (AM)     88.4%        OCCUPIED
433.0 MHz     ISM (BPSK)       82.2%        OCCUPIED

============================================================
```

When a channel is classified as **VACANT**, the system identifies it as a potential spectrum opportunity.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/PragyamSrivastava/Cognitive-Radio-Networks.git
cd Cognitive-Radio-Networks
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard

```bash
python -m streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

### Dashboard Features

The Streamlit dashboard provides:

* **Live Spectrum Scanner**
* **Multi-Channel White-Space Map**
* **Cooperative Fusion Center**
* **Model Analytics**
* **ROC Performance Curves**
* **Spectrum Occupancy Visualization**
* **Phase 2 Advanced Systems**

---

## 🧪 Run Simulation and Train CNN

To train the model and run cooperative sensing simulations:

```bash
python main.py --epochs 10 --coop-trials 10
```

You can modify the number of epochs and cooperative trials according to your computational resources.

---

## 📡 Capture Live RTL-SDR Signals

Connect the RTL-SDR hardware and run:

```bash
python hardware_capture.py
```

The hardware capture module can be used for real-time I/Q signal acquisition and spectrum analysis.

---

## 📁 Repository Structure

```text
Cognitive-Radio-Networks/
│
├── config.py
│
├── signal_generator.py
│
├── spectrogram_processor.py
│
├── cnn_model.py
│
├── fusion_center.py
│
├── phase2_engine.py
│
├── train.py
│
├── evaluate.py
│
├── hardware_capture.py
│
├── dashboard.py
│
├── generate_presentation.py
│
├── generate_poster.py
│
├── presentation.md
│
├── requirements.txt
│
├── .gitignore
│
└── results/
    ├── roc_cooperative.png
    ├── roc_individual.png
    ├── accuracy_vs_snr.png
    ├── confusion_matrices.png
    ├── sample_spectrograms.png
    ├── training_history.png
    ├── metrics_summary.txt
    └── best_model.pth
```

### File Descriptions

| File                       | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `config.py`                | Central configuration and hyperparameters             |
| `signal_generator.py`      | AM, FM, BPSK, QPSK signal generation and AWGN channel |
| `spectrogram_processor.py` | STFT processing and spectrogram generation            |
| `cnn_model.py`             | PyTorch 4-block CNN architecture                      |
| `fusion_center.py`         | Cooperative sensing and fusion logic                  |
| `phase2_engine.py`         | Advanced cooperative sensing and security features    |
| `train.py`                 | Model training pipeline                               |
| `evaluate.py`              | Model evaluation and performance analysis             |
| `hardware_capture.py`      | RTL-SDR hardware signal capture                       |
| `dashboard.py`             | Interactive Streamlit dashboard                       |
| `requirements.txt`         | Python dependencies                                   |

---

## 🛠️ Technologies Used

### Software

* Python
* PyTorch
* Streamlit
* NumPy
* Pandas
* SciPy
* Matplotlib
* Scikit-learn

### Signal Processing

* I/Q Signal Processing
* STFT
* Spectrogram Generation
* AWGN Channel Modeling
* SNR Analysis
* Spectrum Sensing

### Machine Learning

* Convolutional Neural Networks
* Modulation Classification
* Classification Metrics
* ROC Analysis
* Confusion Matrix
* Cooperative Decision Fusion

### Hardware

* RTL-SDR R820T2
* RF Antenna
* Real-Time I/Q Signal Acquisition

---

## 🎯 Project Objectives

The main objectives of this project are:

1. To develop an intelligent spectrum sensing system for Cognitive Radio Networks.
2. To convert raw I/Q signals into 2D spectrogram representations using STFT.
3. To classify wireless signals using a deep CNN.
4. To detect Primary User activity under different SNR conditions.
5. To improve sensing reliability using multiple cooperative sensing nodes.
6. To compare OR Rule and Majority Voting fusion techniques.
7. To integrate the system with RTL-SDR hardware for real-world spectrum sensing.
8. To provide an interactive Streamlit-based visualization and analysis platform.

---

## 🌐 Applications

The proposed system can be useful in:

* Cognitive Radio Networks
* Dynamic Spectrum Access
* Wireless Spectrum Monitoring
* Intelligent Spectrum Sensing
* RF Signal Classification
* Spectrum Occupancy Detection
* IoT Communication
* Software Defined Radio
* Wireless Network Optimization
* Real-Time RF Monitoring

---

## 🔮 Future Scope

Possible future improvements include:

* Integration with additional modulation schemes.
* Support for more cooperative sensing nodes.
* Federated learning for distributed spectrum sensing.
* Advanced deep-learning architectures.
* Real-time spectrum prediction.
* Reinforcement-learning-based channel selection.
* Improved RF hardware integration.
* Detection of malicious sensing nodes.
* Edge deployment on low-power computing platforms.
* Autonomous dynamic spectrum access.

---

## 👥 Project Team & Credits

**Institution:**
Department of Electronics & Telecommunication Engineering
**Bangalore Institute of Technology, Bengaluru, India**

**Project Guide:**
Prof. Asha R

**Team Members:**

* **Pragyam Srivastava** — `1BI23ET041`
* **Sangeetha M** — `1BI23ET050`
* **Rachana M N** — `1BI23ET064`
* **Amoghapriya R B** — `1BI23ET066`

---

## 📜 References

1. I. F. Akyildiz, W.-Y. Lee, M. C. Vuran, and S. Mohanty, *A Survey on Spectrum Management in Cognitive Radio Networks*, IEEE Communications Magazine, vol. 46, no. 4, pp. 40–48, 2008.

2. J. Mitola and G. Q. Maguire, *Cognitive Radio: Making Software Radios More Personal*, IEEE Personal Communications, vol. 6, no. 4, pp. 13–18, 1999.

3. T. J. O'Shea and J. Hoydis, *An Introduction to Deep Learning for the Physical Layer*, IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 4, pp. 563–575, 2017.

4. Y. Sun, J. Wang, and Z. Han, *Deep Learning-Based Cooperative Spectrum Sensing in Cognitive Radio Networks*, IEEE Transactions on Wireless Communications, vol. 18, no. 6, pp. 3126–3139, 2019.

5. S. Rajendran, W. Meert, V. Lenders, and S. Pollin, *Deep Learning Models for Wireless Signal Classification with Distributed Low-Cost Spectrum Sensors*, IEEE Transactions on Cognitive Communications and Networking, vol. 4, no. 3, pp. 433–445, 2018.

---

<div align="center">

### 📡 Cognitive Radio Networks × AI × SDR

**Made with ❤️ by Team G10**
**Department of Electronics & Telecommunication Engineering**
**Bangalore Institute of Technology**

</div>
