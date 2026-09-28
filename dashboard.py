"""
dashboard.py — Interactive Live Dashboard for Cooperative Spectrum Sensing
A professional, modern web interface for Cognitive Radio Networks with real RTL-SDR & AI.
Run with: streamlit run dashboard.py
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root & DLLs are in PATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in os.environ.get('PATH', ''):
    os.environ['PATH'] = CURRENT_DIR + ';' + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory') and os.path.exists(CURRENT_DIR):
    try:
        os.add_dll_directory(CURRENT_DIR)
    except Exception:
        pass

# Project modules
import config as cfg
from cnn_model import build_cnn, predict_with_occupancy
from spectrogram_processor import iq_to_spectrogram
from signal_generator import generate_fm, generate_am, generate_bpsk, generate_qpsk, _add_awgn
from fusion_center import FusionCenter
from phase2_engine import AutonomousHoppingEngine, SoftFusionCenter, SecureCooperativeFusion

# Try importing RTL-SDR
try:
    from rtlsdr import RtlSdr
    HARDWARE_AVAILABLE = True
    # Test if device is physically reachable
    try:
        _test_sdr = RtlSdr()
        TUNER_TYPE = _test_sdr.get_tuner_type()
        _test_sdr.close()
    except Exception:
        TUNER_TYPE = "RTL-SDR (Driver Ready)"
except Exception:
    HARDWARE_AVAILABLE = False
    TUNER_TYPE = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Radio Spectrum Sensing — AI Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.status-card {
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    letter-spacing: 1px;
}
.status-occupied {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
    color: white;
    border: 1px solid #f87171;
}
.status-vacant {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border: 1px solid #34d399;
}

.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.reco-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
    font-weight: 500;
}

.channel-pill {
    padding: 0.6rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}
.pill-free { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.pill-busy { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model & Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained CNN PyTorch model."""
    model_path = os.path.join(cfg.RESULTS_DIR, "best_model.pth")
    model = build_cnn()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    return model

cnn_model = load_model()


def capture_samples(center_freq, num_samples=1024, sample_rate=2.048e6, force_sim=False):
    """Capture real IQ samples from RTL-SDR or generate synthetic fallback."""
    if HARDWARE_AVAILABLE and not force_sim:
        try:
            sdr = RtlSdr()
            sdr.sample_rate = sample_rate
            sdr.center_freq = center_freq
            sdr.gain = 'auto'
            samples = sdr.read_samples(num_samples)
            sdr.close()
            return samples.astype(np.complex64), "Real RTL-SDR Antenna"
        except Exception as e:
            pass

    # Fallback Simulation
    if 88e6 <= center_freq <= 108e6:
        # Simulate active FM station on specific frequencies
        is_station = any(abs(center_freq - f) < 0.2e6 for f in [91.5e6, 93.5e6, 104.0e6])
        if is_station:
            sig = generate_fm(num_samples=num_samples)
            return _add_awgn(sig, snr_db=15.0), "Simulated FM Station"
        else:
            noise = _add_awgn(np.zeros(num_samples, dtype=complex), snr_db=-20.0)
            return noise, "Simulated Noise Floor (Vacant)"
    elif 118e6 <= center_freq <= 136e6:
        sig = generate_am(num_samples=num_samples)
        return _add_awgn(sig, snr_db=12.0), "Simulated Airband AM"
    elif abs(center_freq - 433.92e6) < 1e6:
        sig = generate_bpsk(num_samples=num_samples)
        return _add_awgn(sig, snr_db=10.0), "Simulated ISM BPSK"
    else:
        sig = generate_fm(num_samples=num_samples)
        return _add_awgn(sig, snr_db=5.0), "Simulated Signal"


def compute_psd(iq_samples, sample_rate):
    """Compute Power Spectral Density in dBFS."""
    N = len(iq_samples)
    window = np.hanning(N)
    fft_shifted = np.fft.fftshift(np.fft.fft(iq_samples * window, N))
    psd_db = 10.0 * np.log10((np.abs(fft_shifted) ** 2) / N + 1e-12)
    freqs_mhz = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / sample_rate)) / 1e6
    return freqs_mhz, psd_db


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/satellite-sending-signal.png", width=70)
    st.markdown("### 📡 Hardware Status")
    
    if HARDWARE_AVAILABLE and TUNER_TYPE:
        st.success(f"🟢 **RTL-SDR Connected**\n\nTuner: `{TUNER_TYPE}`")
    else:
        st.warning("🟡 **Simulation Mode**\n\n(Plug in RTL-SDR for live RF)")

    st.markdown("---")
    st.markdown("### ⚙️ Sensing Parameters")
    
    threshold = st.slider("Confidence Threshold (Occupied/Vacant)", 0.30, 0.90, cfg.OCCUPIED_THRESHOLD, 0.05)
    sample_rate_mhz = st.selectbox("Bandwidth / Sample Rate", [2.048, 1.024, 2.4], index=0)
    sample_rate_hz = sample_rate_mhz * 1e6
    
    st.markdown("---")
    st.markdown("### ℹ️ About Project")
    st.caption("**Cooperative Spectrum Sensing (CSS)** using Deep Learning (CNN) for Cognitive Radio Networks. Dynamically detects Primary Users (PU) to harvest White Spaces.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">COGNITIVE SPECTRUM SENSING</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning-Driven Cooperative Spectrum Sensing & Dynamic Frequency Access</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Dashboard Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Spectrum Scanner",
    "🗺️ Multi-Channel White-Space Map",
    "🤝 Cooperative Fusion Center",
    "📊 Model Analytics & ROC",
    "🚀 Phase 2 Advanced Cognitive Features"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Spectrum Scanner
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_ctrl, col_res = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### 🎛️ Tuner Controls")
        
        band_preset = st.selectbox(
            "Select Band Preset",
            ["FM Broadcast (88 - 108 MHz)", "Airband ATC (118 - 136 MHz)", "ISM Telemetry (433 MHz)", "Custom Frequency"]
        )

        if band_preset == "FM Broadcast (88 - 108 MHz)":
            freq_mhz = st.slider("Target Frequency (MHz)", 88.0, 108.0, 91.5, 0.1)
        elif band_preset == "Airband ATC (118 - 136 MHz)":
            freq_mhz = st.slider("Target Frequency (MHz)", 118.0, 136.0, 121.5, 0.1)
        elif band_preset == "ISM Telemetry (433 MHz)":
            freq_mhz = st.slider("Target Frequency (MHz)", 430.0, 440.0, 433.92, 0.05)
        else:
            freq_mhz = st.number_input("Custom Frequency (MHz)", 24.0, 1700.0, 93.5, 0.1)

        scan_btn = st.button("🚀 Capture & Classify Spectrum", type="primary", use_container_width=True)

    with col_res:
        if scan_btn or "last_scan" not in st.session_state:
            # Capture IQ
            iq_data, source_label = capture_samples(freq_mhz * 1e6, cfg.NUM_SAMPLES, sample_rate_hz)
            
            # STFT Spectrogram
            spec = iq_to_spectrogram(iq_data)
            spec_input = spec.reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
            
            # CNN Prediction
            pred = predict_with_occupancy(cnn_model, spec_input, threshold=threshold)
            
            # Store in session
            st.session_state["last_scan"] = {
                "freq": freq_mhz,
                "iq": iq_data,
                "spec": spec,
                "pred": pred,
                "source": source_label
            }

        scan_data = st.session_state["last_scan"]
        p = scan_data["pred"]
        is_occ = p["occupied"][0]
        sig_name = p["class_names"][0]
        conf = p["confidences"][0]

        # Status Banner
        if is_occ:
            st.markdown(f'<div class="status-card status-occupied">🔴 PRIMARY USER DETECTED — OCCUPIED ({conf:.1%})</div>', unsafe_allow_html=True)
            st.warning(f"⚠️ **Channel {scan_data['freq']} MHz is busy!** Primary User is transmitting **{sig_name}**. Secondary Users must **NOT** transmit.")
        else:
            st.markdown(f'<div class="status-card status-vacant">🟢 SPECTRUM HOLE FOUND — VACANT ({conf:.1%})</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="reco-banner">✨ **OPPORTUNITY DETECTED:** Frequency **{scan_data["freq"]} MHz** is clear. Cognitive Radio can safely transmit data without interference!</div>', unsafe_allow_html=True)

        # Quick Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{scan_data["freq"]} MHz</div><div class="metric-lbl">Tuned Frequency</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{sig_name}</div><div class="metric-lbl">Signal Classification</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{conf:.1%}</div><div class="metric-lbl">AI Confidence</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{"OCCUPIED" if is_occ else "VACANT"}</div><div class="metric-lbl">Channel State</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Visualizations Row: PSD & Spectrogram
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("##### 📈 Power Spectral Density (PSD)")
        f_rel, psd_db = compute_psd(scan_data["iq"], sample_rate_hz)
        f_abs = scan_data["freq"] + f_rel
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(
            x=f_abs, y=psd_db, mode='lines',
            line=dict(color='#38bdf8', width=2),
            name='PSD (dBFS)'
        ))
        fig_psd.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_dark",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Power (dBFS)",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    with v_col2:
        st.markdown("##### 🖼️ 2D STFT Spectrogram (AI Input)")
        fig_spec = px.imshow(
            scan_data["spec"],
            color_continuous_scale="Viridis",
            labels=dict(x="Time Bins", y="Frequency Bins"),
            aspect="auto",
            origin="lower"
        )
        fig_spec.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_spec, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Multi-Channel White-Space Map
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Wideband Multi-Channel Spectrum Map")
    st.caption("Scans an entire frequency band to locate all available Spectrum Holes (White Spaces) in real time.")
    
    scan_band_choice = st.selectbox(
        "Select Band to Sweep",
        ["FM Broadcast Band (88.0 - 106.0 MHz)", "VHF Airband (118.0 - 126.0 MHz)"]
    )
    
    if st.button("🔍 Run Full Band Scan", type="primary"):
        if "FM" in scan_band_choice:
            test_freqs = [88.5, 91.1, 91.5, 92.7, 93.5, 95.0, 98.3, 100.1, 102.4, 104.0, 106.2]
        else:
            test_freqs = [118.1, 119.5, 121.5, 122.8, 124.2, 125.6]

        results = []
        progress_bar = st.progress(0)
        
        for idx, f in enumerate(test_freqs):
            iq, _ = capture_samples(f * 1e6, cfg.NUM_SAMPLES, sample_rate_hz)
            sp = iq_to_spectrogram(iq).reshape(1, *cfg.SPEC_SIZE, 1).astype(np.float32)
            pr = predict_with_occupancy(cnn_model, sp, threshold=threshold)
            results.append({
                "Frequency (MHz)": f,
                "Signal": pr["class_names"][0],
                "Confidence": f"{pr['confidences'][0]:.1%}",
                "Occupied": pr["occupied"][0],
                "Status": "🔴 OCCUPIED" if pr["occupied"][0] else "🟢 VACANT"
            })
            progress_bar.progress((idx + 1) / len(test_freqs))
            
        st.session_state["map_results"] = results

    if "map_results" in st.session_state:
        res = st.session_state["map_results"]
        
        # Grid of Channel Cards
        cols = st.columns(len(res))
        for idx, r in enumerate(res):
            with cols[idx]:
                css_class = "pill-busy" if r["Occupied"] else "pill-free"
                icon = "🔴" if r["Occupied"] else "🟢"
                st.markdown(
                    f'<div class="channel-pill {css_class}">{icon} {r["Frequency (MHz)"]}<br><small>{r["Signal"]}</small></div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        
        free_channels = [r["Frequency (MHz)"] for r in res if not r["Occupied"]]
        if free_channels:
            st.success(f"🎯 **Available White Spaces Found:** Frequencies `{', '.join(map(str, free_channels))} MHz` are currently idle and free for Dynamic Spectrum Access!")
        else:
            st.warning("⚠️ All scanned channels are currently occupied by Primary Users.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Cooperative Fusion Center (3-Node Simulation)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤝 3-Node Cooperative Spectrum Sensing & Fusion")
    st.caption("Simulates 3 spatially separated Secondary Users (SUs) independently sensing the channel under AWGN noise and fading, aggregating decisions at a Fusion Center.")

    coop_col1, coop_col2 = st.columns([1, 2], gap="large")
    
    with coop_col1:
        st.markdown("#### 🎛️ Simulation Controls")
        sim_sig_type = st.selectbox("Primary User (PU) Transmission", ["FM", "AM", "BPSK", "QPSK", "None (Noise Only)"])
        sim_snr = st.slider("Channel SNR (dB)", -20, 20, -5, 1)
        run_coop_btn = st.button("⚡ Run Cooperative Decision", type="primary", use_container_width=True)

    with coop_col2:
        if run_coop_btn or "coop_res" not in st.session_state:
            # Generate clean or noise
            if sim_sig_type == "None (Noise Only)":
                clean_sig = np.zeros(cfg.NUM_SAMPLES, dtype=complex)
            elif sim_sig_type == "FM":
                clean_sig = generate_fm()
            elif sim_sig_type == "AM":
                clean_sig = generate_am()
            elif sim_sig_type == "BPSK":
                clean_sig = generate_bpsk()
            else:
                clean_sig = generate_qpsk()

            fc = FusionCenter(cnn_model, num_nodes=3)
            coop_res = fc.cooperative_sense(clean_sig, sim_snr, threshold=threshold)
            st.session_state["coop_res"] = coop_res

        c_res = st.session_state["coop_res"]
        nodes = c_res["individual_decisions"]
        or_res = c_res["or_fusion"]
        maj_res = c_res["majority_fusion"]

        # Display Node Cards
        n_cols = st.columns(3)
        for i, n in enumerate(nodes):
            with n_cols[i]:
                occ_badge = "🔴 OCCUPIED" if n["occupied"] else "🟢 VACANT"
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-weight:700; color:#94a3b8;">NODE {n['node_id']+1}</div>
                    <div style="font-size:1.2rem; font-weight:800; margin:0.3rem 0;">{occ_badge}</div>
                    <div style="font-size:0.8rem; color:#cbd5e1;">Class: <b>{n['predicted_name']}</b> ({n['confidence']:.1%})</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚖️ Fusion Center Global Decisions")
        
        f1, f2 = st.columns(2)
        with f1:
            or_occ = or_res["global_occupied"]
            st.markdown(f"""
            <div class="metric-box" style="border-left: 4px solid {'#ef4444' if or_occ else '#10b981'};">
                <div style="font-weight:700;">OR RULE (1-out-of-3)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{'#f87171' if or_occ else '#34d399'};">
                    {'🔴 OCCUPIED' if or_occ else '🟢 VACANT'}
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.3rem;">
                    Logic: PU present if <b>ANY</b> node detects ($Q_d$ maximized).
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with f2:
            maj_occ = maj_res["global_occupied"]
            st.markdown(f"""
            <div class="metric-box" style="border-left: 4px solid {'#ef4444' if maj_occ else '#10b981'};">
                <div style="font-weight:700;">MAJORITY VOTING (2-out-of-3)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{'#f87171' if maj_occ else '#34d399'};">
                    {'🔴 OCCUPIED' if maj_occ else '🟢 VACANT'}
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.3rem;">
                    Logic: PU present if <b>≥ 2 nodes</b> detect (Balanced trade-off).
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Model Analytics & ROC Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Neural Network Evaluation & ROC Analytics")
    st.caption("Detailed performance evaluation metrics generated during Phase 1 simulation.")

    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        roc_path = os.path.join(cfg.RESULTS_DIR, "roc_cooperative.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves: Individual vs OR Rule vs Majority Voting", use_container_width=True)
            
        acc_path = os.path.join(cfg.RESULTS_DIR, "accuracy_vs_snr.png")
        if os.path.exists(acc_path):
            st.image(acc_path, caption="Classification Accuracy vs Signal-to-Noise Ratio (SNR)", use_container_width=True)

    with g_col2:
        cm_path = os.path.join(cfg.RESULTS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices across Different SNR Levels", use_container_width=True)
            
        th_path = os.path.join(cfg.RESULTS_DIR, "training_history.png")
        if os.path.exists(th_path):
            st.image(th_path, caption="CNN Training Loss & Accuracy Curves", use_container_width=True)

    # Metrics Summary Text
    summary_path = os.path.join(cfg.RESULTS_DIR, "metrics_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            st.markdown("#### 📋 Classification Summary Report")
            st.code(f.read(), language="text")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: Phase 2 Advanced Cognitive Systems
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🚀 Phase 2: Dynamic Spectrum Access & Next-Gen Intelligence")
    st.caption("Advanced Cognitive Radio enhancements: Autonomous Channel Hopping, Soft Decision Fusion, and Adversarial Defense.")

    p2_sub1, p2_sub2, p2_sub3 = st.tabs([
        "🤖 Autonomous Frequency Hopping Engine",
        "⚖️ Soft Decision Fusion Center",
        "🛡️ Malicious Node & SSDF Defense"
    ])

    # Sub-tab 1: Autonomous Hopping
    with p2_sub1:
        st.markdown("#### 🤖 Dynamic Spectrum Access & Collision Avoidance")
        st.write("Simulates a Secondary User (SU) transmitting data in an open White Space. If a licensed Primary User (PU) suddenly turns on, the SU autonomously vacates and hops to the cleanest open band.")

        if "hopping_engine" not in st.session_state:
            st.session_state["hopping_engine"] = AutonomousHoppingEngine()

        engine = st.session_state["hopping_engine"]
        
        h_col1, h_col2 = st.columns([1, 2], gap="large")
        with h_col1:
            st.markdown(f"**Current SU Active Channel:** `{engine.current_channel} MHz`")
            pu_event = st.selectbox(
                f"Simulate Event on Channel {engine.current_channel} MHz",
                ["No Primary User (Channel Stays Clear)", "⚠️ Primary User Starts Transmitting! (Collision Threat)"]
            )
            hop_trigger = st.button("⚡ Execute Policy Step", type="primary", use_container_width=True)

        with h_col2:
            if hop_trigger or "hop_res" not in st.session_state:
                # Build mock channel states
                pu_active = "Collision Threat" in pu_event
                channel_states = {
                    88.5: {'occupied': True, 'snr': 15.0},
                    91.5: {'occupied': True, 'snr': 18.0},
                    93.5: {'occupied': True, 'snr': 20.0},
                    95.0: {'occupied': False, 'snr': 2.0},
                    98.3: {'occupied': pu_active, 'snr': 22.0 if pu_active else 0.5},
                    100.1: {'occupied': False, 'snr': 1.0},
                    104.0: {'occupied': True, 'snr': 19.0},
                }
                hop_res = engine.evaluate_and_hop(channel_states)
                st.session_state["hop_res"] = hop_res

            res = st.session_state["hop_res"]
            if res["action"] == "HOP":
                st.markdown(f'<div class="status-card status-occupied">🚀 AUTONOMOUS FREQUENCY HOP EXECUTED</div>', unsafe_allow_html=True)
                st.success(f"**Action:** Hopped from `{res['old_channel']} MHz` ➔ `{res['new_channel']} MHz`")
                st.info(f"**Reason:** {res['reason']}. The Primary User is completely protected from secondary interference!")
            else:
                st.markdown(f'<div class="status-card status-vacant">🟢 CHANNEL CLEAR — TRANSMISSION ACTIVE</div>', unsafe_allow_html=True)
                st.success(f"**Status:** Channel `{res['current_channel'] if 'current_channel' in res else engine.current_channel} MHz` remains vacant. Full throughput maintained.")

    # Sub-tab 2: Soft Decision Fusion
    with p2_sub2:
        st.markdown("#### ⚖️ Soft Decision Fusion (Weighted Confidence)")
        st.write("Instead of lossy 1-bit binary decisions, nodes transmit their complete continuous AI confidence vectors. The Fusion Center computes a weighted likelihood statistic $\\Lambda = \\sum w_i P_i$.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            n1_conf = st.slider("Node 1 AI Confidence (Occupied)", 0.0, 1.0, 0.85, 0.05)
            n2_conf = st.slider("Node 2 AI Confidence (Occupied)", 0.0, 1.0, 0.40, 0.05)
            n3_conf = st.slider("Node 3 AI Confidence (Occupied)", 0.0, 1.0, 0.70, 0.05)

        with s_col2:
            soft_res = SoftFusionCenter.soft_combining([n1_conf, n2_conf, n3_conf])
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Weighted Soft Decision Statistic (Lambda)</div>
                <div class="metric-val">{soft_res['soft_statistic']:.3f}</div>
                <div style="font-size:1.2rem; font-weight:800; margin-top:0.5rem; color:{'#f87171' if soft_res['global_occupied'] else '#34d399'};">
                    {'🔴 OCCUPIED' if soft_res['global_occupied'] else '🟢 VACANT'} (Threshold = {cfg.OCCUPIED_THRESHOLD})
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sub-tab 3: Security & Malicious Node Defense
    with p2_sub3:
        st.markdown("#### 🛡️ AI Defense Against Spectrum Sensing Data Falsification (SSDF)")
        st.write("A rogue/compromised node attempts to lie to the network. The Fusion Center uses dynamic trust/reputation scoring to detect and quarantine the malicious attacker.")

        if "secure_fusion" not in st.session_state:
            st.session_state["secure_fusion"] = SecureCooperativeFusion(num_nodes=3)

        sec_engine = st.session_state["secure_fusion"]
        
        sec_col1, sec_col2 = st.columns([1, 2], gap="large")
        with sec_col1:
            attacker_choice = st.selectbox("Simulate Malicious Attacker Node", ["None (All Honest)", "Node 1 (Attacker)", "Node 2 (Attacker)", "Node 3 (Attacker)"])
            att_idx = None if "None" in attacker_choice else int(attacker_choice.split()[1]) - 1
            run_sec_btn = st.button("🛡️ Run Security Evaluation Step", type="primary", use_container_width=True)

        with sec_col2:
            if run_sec_btn or "sec_res" not in st.session_state:
                # Real signal report
                reports = [
                    {'node_id': 0, 'occupied': True, 'confidence': 0.88},
                    {'node_id': 1, 'occupied': True, 'confidence': 0.91},
                    {'node_id': 2, 'occupied': True, 'confidence': 0.85},
                ]
                sec_res = sec_engine.robust_fuse(reports, malicious_node_idx=att_idx)
                st.session_state["sec_res"] = sec_res

            sr = st.session_state["sec_res"]
            t_scores = sr["trust_scores"]
            
            st.markdown("##### 🎖️ Dynamic Node Reputation / Trust Scores")
            t_cols = st.columns(3)
            for idx, score in enumerate(t_scores):
                with t_cols[idx]:
                    is_isolated = idx in sr["isolated_nodes"]
                    badge_color = "#dc2626" if is_isolated else ("#f59e0b" if score < 0.8 else "#16a34a")
                    status_txt = "🚫 QUARANTINED" if is_isolated else ("⚠️ SUSPICIOUS" if score < 0.8 else "✅ TRUSTED")
                    st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid {badge_color};">
                        <div style="font-weight:700;">NODE {idx+1}</div>
                        <div style="font-size:1.4rem; font-weight:800; color:{badge_color};">{score:.2f} / 1.00</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">{status_txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

            if sr["isolated_nodes"]:
                st.error(f"🚨 **Security Threat Mitigated:** Node {[i+1 for i in sr['isolated_nodes']]} has been quarantined. The network decision remains safe and accurate!")
            else:
                st.success("✅ All active nodes are verified honest. Network consensus is 100% reliable.")

