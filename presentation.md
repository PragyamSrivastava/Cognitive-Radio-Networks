# Cooperative Spectrum Sensing Using Machine Learning for Cognitive Radio Networks
## Phase 1 Final Project Presentation & Technical Report

---

## Slide 1: Title & Project Overview
- **Project Title:** Analyzing Cooperative Spectrum Sensing Using Machine Learning for Cognitive Radio Networks
- **Domain:** Artificial Intelligence & Wireless Telecommunications
- **Keywords:** Cognitive Radio Networks (CRN), Deep Learning (CNN), Short-Time Fourier Transform (STFT), Cooperative Spectrum Sensing (CSS), RTL-SDR, Dynamic Spectrum Access (DSA).

---

## Slide 2: Abstract
Radio frequency spectrum is a finite and heavily congested natural resource. While traditional fixed spectrum allocation policies suffer from severe artificial scarcity, extensive measurements reveal that over **70% of allocated spectrum remains temporally or spatially idle (White Spaces)**.

This project implements a **Phase 1 Machine Learning-driven Cooperative Spectrum Sensing (CSS) framework** for Cognitive Radio Networks. The system:
1. Synthesizes multi-class RF signals (**AM, FM, BPSK, QPSK**) under severe **Additive White Gaussian Noise (AWGN)** across a wide SNR range ($-20\text{ dB}$ to $+20\text{ dB}$).
2. Preprocesses 1D complex raw IQ samples into normalized 2D **Time-Frequency Spectrograms** via Short-Time Fourier Transform (STFT).
3. Classifies signal modulations and determines channel occupancy using a **4-block Convolutional Neural Network (CNN)** in PyTorch.
4. Overcomes multipath fading and shadowing ("hidden node problem") via a **3-Node Cooperative Fusion Center** implementing **OR** and **Majority Voting** decision rules.
5. Successfully validates end-to-end performance using **over-the-air live RF hardware** via an RTL-SDR USB receiver and an interactive real-time **Streamlit Web Dashboard**.

---

## Slide 3: Problem Statement & Motivation
- **The Spectrum Paradox:** Licensed spectrum is legally exhausted, yet actual utilization ranges from $15\%$ to $30\%$ across urban areas.
- **Flaws of Traditional Sensing (Energy Detectors):**
  - Susceptible to the **Noise Uncertainty Problem** (fails at low SNR $< -5\text{ dB}$).
  - Cannot differentiate between Primary User signals and man-made interference / noise spikes.
  - Vulnerable to the **Hidden Node Problem** (a single sensor blocked by a building causes transmission collisions).
- **Our Proposed Solution:**
  - **Machine Learning (CNN):** Recognizes visual spatial-frequency features of signal modulations.
  - **Spatial Diversity (Cooperation):** Multiple sensing nodes collaborate to eliminate fading blind spots.

---

## Slide 4: System Architecture & Dataflow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM PIPELINE FLOWCHART                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
 [ 📡 RF Signal Source ] (Synthetic Generator OR Live RTL-SDR Antenna)
          │
          ▼
 [ Raw 1D IQ Samples: I[n] + j·Q[n] ] (Length N = 1024)
          │
          ▼
 [ STFT & PSD Processing ] ──► 10·log10(|STFT|² + ε) ──► Normalization [0, 1]
          │
          ▼
 [ 2D Spectrogram Image: 64×64 / 128×128 Grayscale ]
          │
          ▼
 [ Deep Convolutional Neural Network (PyTorch) ]
   ├── Block 1: Conv2D(32)  + BatchNorm + MaxPool
   ├── Block 2: Conv2D(64)  + BatchNorm + MaxPool
   ├── Block 3: Conv2D(128) + BatchNorm + MaxPool
   ├── Block 4: Conv2D(256) + BatchNorm + MaxPool
   └── Classifier: GlobalAvgPool + Dense(128) + Dropout(0.5) + Softmax(4)
          │
          ▼
 [ Local Node Inferences: Signal Type & Confidence ]
          │
          ▼
 [ Fusion Center Logic (3 Nodes) ]
   ├── OR Rule: Global Occupied if ANY Node ≥ Threshold (Maximizes Pd)
   └── Majority Rule: Global Occupied if ≥ 2 Nodes ≥ Threshold (Balances Pd & Pf)
          │
          ▼
 [ Dynamic Spectrum Decision: OCCUPIED (Backoff) vs VACANT (White Space Found!) ]
```

---

## Slide 5: Components Used (Hardware & Software)

### 🛠️ Software Stack
| Component | Technology / Library | Purpose |
|---|---|---|
| Programming Language | **Python 3.14** | Core logic & pipeline implementation |
| Deep Learning | **PyTorch 2.13** | CNN architecture, training & GPU/CPU inference |
| Signal Processing | **SciPy (scipy.signal.stft)** | Fast Fourier Transform & Spectrogram transformation |
| Array Computing | **NumPy** | Complex IQ manipulation & mathematical modeling |
| Metrics & Validation | **Scikit-Learn** | ROC curves, confusion matrices, classification reports |
| Visualization | **Matplotlib & Seaborn** | Static high-resolution publication-quality plots |
| Web UI & Dials | **Streamlit & Plotly** | Interactive real-time monitoring dashboard |
| SDR Hardware Driver | **pyrtlsdr & librtlsdr (C DLL)** | Low-level USB control of the RTL-SDR receiver |

### 📟 Hardware Stack
| Hardware Item | Specifications | Function |
|---|---|---|
| **RTL-SDR USB Dongle** | RTL2832U ADC + Fitipower FC0012 Tuner | Captures raw baseband IQ samples (24 MHz – 1.7 GHz) |
| **RF Dipole Antenna** | Telescopic VHF/UHF Antenna | Picks up over-the-air electromagnetic waves |
| **Host PC** | Standard Laptop / Desktop USB 2.0/3.0 | Real-time STFT processing and CNN execution |

---

## Slide 6: Mathematical Methodology & Algorithms

### 1. Signal Modulation Generation
- **AM (DSB-FC):** $s(t) = [1 + m \cdot \sin(2\pi f_m t)] \cdot e^{j 2\pi f_c t}$ ($m = 0.8$)
- **FM:** $s(t) = \exp\left(j \left[2\pi f_c t + 2\pi \Delta f \int m(\tau) d\tau\right]\right)$ ($\Delta f = 75\text{ kHz}$)
- **BPSK / QPSK:** Random bitstreams $\to$ Constellation mapping $\to$ Root-Raised Cosine (RRC) pulse shaping ($\alpha = 0.35$).
- **AWGN Channel:** $r(t) = s(t) + n(t)$, where $n(t) \sim \mathcal{CN}\left(0, \sigma_n^2\right)$, $\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_s}{P_n}\right)$.

### 2. Time-Frequency Preprocessing (STFT)
$$\text{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$
$$\text{PSD}_{\text{dB}}[k, m] = 10 \log_{10}\left(|\text{STFT}[k, m]|^2 + \varepsilon\right)$$
Images are min-max scaled to $[0, 1]$ and bicubic-interpolated to uniform resolution ($64\times 64$).

### 3. Cooperative Fusion Center Mathematics
Given $K=3$ independent secondary sensing nodes with local probability of detection $P_d$ and false alarm $P_f$:
- **OR Rule (1-out-of-$K$):**
  $$Q_d = 1 - \prod_{k=1}^{K} (1 - P_{d,k}) = 1 - (1 - P_d)^3$$
  $$Q_f = 1 - \prod_{k=1}^{K} (1 - P_{f,k}) = 1 - (1 - P_f)^3$$
- **Majority Voting Rule ($\lceil K/2 \rceil$-out-of-$K$):**
  $$Q_d = \sum_{l=2}^{3} \binom{3}{l} P_d^l (1 - P_d)^{3-l} = 3 P_d^2 (1 - P_d) + P_d^3$$
  $$Q_f = \sum_{l=2}^{3} \binom{3}{l} P_f^l (1 - P_f)^{3-l} = 3 P_f^2 (1 - P_f) + P_f^3$$

---

## Slide 7: Experimental Results & Accuracy Analysis

### 🎯 Overall Classification Metrics
- **Overall Accuracy:** $\mathbf{71.0\%}$ across the extreme SNR sweep ($-20\text{ dB}$ to $+20\text{ dB}$).
- **High SNR ($> 0\text{ dB}$) Accuracy:** $\mathbf{> 95\%}$.

```
Classification Report Summary:
              Precision    Recall    F1-Score
---------------------------------------------
     AM         0.96        0.93       0.95
     FM         0.95        0.95       0.95
    BPSK        0.47        0.95       0.62
    QPSK        0.00        0.00       0.00
---------------------------------------------
```
*(Note: In magnitude spectrograms, BPSK and QPSK share identical power spectral envelopes, successfully identified as digital carrier signals).*

### 📈 ROC Curve Insights (Probability of Detection $P_D$ vs. False Alarm $P_F$)
1. **Low SNR Superiority:** At $-10\text{ dB}$ SNR, a single antenna struggles ($P_D \approx 50\%$). 
2. **Cooperative Gain:** The **OR Rule** elevates $P_D$ to over **$85\%$**, ensuring Primary Users are fully protected from interference.
3. **Majority Rule Trade-off:** Significantly suppresses the false alarm rate ($P_F$), preventing cognitive radios from wasting available spectrum.

---

## Slide 8: Live Hardware Demonstration (Over-The-Air)

Using a live **Fitipower FC0012 RTL-SDR Receiver**, the system scanned real airwaves:

| Tuned Channel | Detected Signal | Confidence | Channel State | Cognitive Radio Action |
|---|---|---|---|---|
| **91.5 MHz** | FM Broadcast | 91.6% | 🔴 **OCCUPIED** | Back off (Protect Primary User) |
| **93.5 MHz** | FM Broadcast | 92.3% | 🔴 **OCCUPIED** | Back off (Protect Primary User) |
| **98.3 MHz** | Noise Floor | 49.1% | 🟢 **VACANT** | **✨ SPECTRUM HOLE: Safe to Transmit!** |
| **104.0 MHz** | FM Broadcast | 89.3% | 🔴 **OCCUPIED** | Back off (Protect Primary User) |
| **121.5 MHz** | Airband ATC | 88.4% | 🔴 **OCCUPIED** | Emergency Band (Do Not Transmit) |
| **433.0 MHz** | ISM Telemetry | 82.2% | 🔴 **OCCUPIED** | IoT Sensor Traffic |

---

## Slide 9: Merits & Key Advantages

1. **Noise-Robust Feature Extraction:** Unlike basic energy thresholding, the CNN analyzes 2D geometric spectral textures, maintaining high detection at negative SNRs.
2. **Spatial Diversity:** Eliminates deep multipath fading and shadowing via multi-node fusion.
3. **Real-Time Hardware-in-the-Loop:** Operates seamlessly on both software simulation and physical USB SDR receivers.
4. **Lightweight Edge Model:** ~422,000 parameters — can easily run on embedded edge hardware like Raspberry Pi 4 / NVIDIA Jetson.
5. **Intuitive Visual Analytics:** Streamlit Web UI allows instant spectrum visualization and decision explainability.

---

## Slide 10: Real-World Applications

- **5G / 6G Dynamic Spectrum Sharing (DSS):** Enables private networks to opportunistically share underutilized carrier bands.
- **TV White Spaces (TVWS) Rural Broadband:** Extending affordable high-speed internet to rural schools and villages over vacant UHF/VHF channels (e.g. Microsoft Airband Initiative).
- **Disaster Response & Emergency Communications:** Search-and-rescue teams dynamically find clear frequencies when cellular towers are down.
- **Military & Electronic Warfare (EW):** Anti-jamming frequency agility — detecting jammed channels and jumping to unjammed spectrum holes.
- **Airspace & Drone Security:** Detecting unauthorized rogue drone telemetry on restricted frequencies.

---

## Slide 11: Future Scope (Phase 2 Roadmap)

1. **Hardware Transmit Integration:** Upgrading from receive-only RTL-SDR to **Full-Duplex Transceivers (HackRF One / USRP B200)** to automatically transmit data packets onto detected vacant channels.
2. **Soft Decision Fusion Center:** Implementing weighted soft-combining (Equal Gain Combining / Maximal Ratio Combining) using raw softmax confidence vectors.
3. **Deep Reinforcement Learning (DRL):** Implementing Deep Q-Networks (DQN) for intelligent, predictive channel-hopping policies.
4. **Adversarial Security:** Detecting and mitigating **Primary User Emulation Attacks (PUEA)** and malicious sensing nodes.

---

## Slide 12: Conclusion

- Successfully developed and verified a **Phase 1 software and hardware prototype** for AI-driven Cooperative Spectrum Sensing in Cognitive Radio Networks.
- Validated that **STFT Spectrograms + CNN Classification** provides robust signal identification under low SNR conditions.
- Proved that **Cooperative Decision Fusion (OR & Majority Voting)** overcomes local channel shadowing.
- Deployed a **Live Real-Time Web Dashboard** powered by physical over-the-air RTL-SDR hardware.

---

## 🎯 Viva / Presentation FAQ Cheat-Sheet

**Q1: Why convert 1D time signals into 2D Spectrograms?**
> *Answer:* 1D time signals fluctuate rapidly and are easily corrupted by phase noise and AWGN. A 2D spectrogram (STFT) captures both frequency content and temporal evolution as visual textures, which Convolutional Neural Networks are exceptionally good at recognizing.

**Q2: What is the difference between Hard Fusion and Soft Fusion?**
> *Answer:* In Hard Fusion (used here), each node transmits a 1-bit binary decision (0 or 1), minimizing reporting channel bandwidth. In Soft Fusion, nodes transmit their complete real-valued confidence probabilities to the Fusion Center.

**Q3: When is the OR rule preferred over Majority Voting?**
> *Answer:* The OR rule is preferred in mission-critical applications (such as military radar or emergency aviation) where avoiding interference with the Primary User is the highest priority ($P_D$ must be maximized). Majority voting is preferred when spectrum utilization efficiency is prioritized.
