"""
generate_poster.py — Generates a print-ready A3 Academic Project Presentation Poster
for Bangalore Institute of Technology matching the exact template grid layout.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
OUTPUT_PPTX = os.path.join(CURRENT_DIR, "A3_Project_Poster_BIT.pptx")
DESKTOP_PPTX = os.path.join(os.path.expanduser("~"), "Desktop", "A3_Project_Poster_BIT.pptx")

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette (BIT Academic Green, Gold, Navy & Crisp White)
# ─────────────────────────────────────────────────────────────────────────────
BIT_GREEN = RGBColor(16, 107, 60)      # #106b3c
BIT_GOLD = RGBColor(217, 119, 6)       # #d97706
BORDER_GRAY = RGBColor(30, 41, 59)     # Dark Slate #1e293b
BG_LIGHT = RGBColor(248, 250, 252)     # #f8fafc
CARD_WHITE = RGBColor(255, 255, 255)
TEXT_MAIN = RGBColor(15, 23, 42)       # #0f172a
TEXT_MUTED = RGBColor(71, 85, 105)     # #475569
HEADER_BG = RGBColor(240, 253, 244)    # Light Green Tint


def create_poster():
    prs = Presentation()
    # A3 Landscape dimensions: 420mm x 297mm = 16.535 in x 11.693 in
    prs.slide_width = Inches(16.535)
    prs.slide_height = Inches(11.693)
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # 1. Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_LIGHT
    bg.line.fill.background()

    # Outer Poster Border
    outer_border = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(0.25), Inches(16.035), Inches(11.193)
    )
    outer_border.fill.background()
    outer_border.line.color.rgb = BIT_GREEN
    outer_border.line.width = Pt(3)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Institutional Header (Bangalore Institute of Technology)
    # ─────────────────────────────────────────────────────────────────────────
    inst_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(15.335), Inches(0.8))
    tf = inst_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "BANGALORE INSTITUTE OF TECHNOLOGY"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = BIT_GREEN
    p.alignment = PP_ALIGN.CENTER

    p_sub = tf.add_paragraph()
    p_sub.text = "Institution of Rajya Vokkaligara Sangha  |  An Autonomous Institution under VTU  |  Dept. of Electronics & Telecommunication Engineering"
    p_sub.font.size = Pt(11)
    p_sub.font.bold = True
    p_sub.font.color.rgb = TEXT_MUTED
    p_sub.alignment = PP_ALIGN.CENTER

    # "PROJECT PRESENTATION" Banner Badge
    banner = slide.shapes.add_shape(
        MSO_SHAPE.PENTAGON, Inches(5.5), Inches(1.2), Inches(5.535), Inches(0.45)
    )
    banner.fill.solid(); banner.fill.fore_color.rgb = BIT_GREEN
    banner.line.fill.background()
    tf_b = banner.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "PROJECT POSTER PRESENTATION"
    p_b.font.size = Pt(13)
    p_b.font.bold = True
    p_b.font.color.rgb = CARD_WHITE
    p_b.alignment = PP_ALIGN.CENTER

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Project Title & Metadata Box
    # ─────────────────────────────────────────────────────────────────────────
    # Title Box
    title_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.75), Inches(15.335), Inches(0.85)
    )
    title_box.fill.solid(); title_box.fill.fore_color.rgb = CARD_WHITE
    title_box.line.color.rgb = BORDER_GRAY
    title_box.line.width = Pt(1.5)
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "Title: Analyzing Cooperative Spectrum Sensing Using Machine Learning for Cognitive Radio Networks"
    p_t.font.size = Pt(15)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_MAIN

    p_t_sub = tf_t.add_paragraph()
    p_t_sub.text = "Focus: AI-Driven Spectrogram Classification (CNN), Spatial Decision Fusion (OR / Majority Rules) & Real RTL-SDR Hardware Validation"
    p_t_sub.font.size = Pt(10.5)
    p_t_sub.font.color.rgb = BIT_GREEN
    p_t_sub.font.bold = True

    # Project ID Box (Left)
    id_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(2.68), Inches(4.2), Inches(0.55)
    )
    id_box.fill.solid(); id_box.fill.fore_color.rgb = CARD_WHITE
    id_box.line.color.rgb = BORDER_GRAY
    id_box.line.width = Pt(1.5)
    tf_id = id_box.text_frame
    p_id = tf_id.paragraphs[0]
    p_id.text = "Project ID: G10"
    p_id.font.size = Pt(13)
    p_id.font.bold = True
    p_id.font.color.rgb = BIT_GOLD

    # Team & Guide Box (Right)
    team_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.9), Inches(2.68), Inches(11.035), Inches(0.55)
    )
    team_box.fill.solid(); team_box.fill.fore_color.rgb = CARD_WHITE
    team_box.line.color.rgb = BORDER_GRAY
    team_box.line.width = Pt(1.5)
    tf_tm = team_box.text_frame
    p_tm = tf_tm.paragraphs[0]
    p_tm.text = "Team Members: Sangeetha, Pragyam Srivastava, Amoghapriya R B, Rachana M N  |  Guide: Prof. Asha R"
    p_tm.font.size = Pt(12)
    p_tm.font.bold = True
    p_tm.font.color.rgb = TEXT_MAIN

    # ─────────────────────────────────────────────────────────────────────────
    # 4. The 6 Main Content Grid Boxes (3 Columns x 2 Rows)
    # ─────────────────────────────────────────────────────────────────────────
    col_w = Inches(4.95)
    col_gap = Inches(0.24)
    x1 = Inches(0.6)
    x2 = x1 + col_w + col_gap
    x3 = x2 + col_w + col_gap
    
    row1_y = Inches(3.35)
    row1_h = Inches(3.55)
    
    row2_y = Inches(7.02)
    row2_h = Inches(3.45)

    def make_card(x, y, w, h, header_title, accent_color=BIT_GREEN):
        # Card Background
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        card.fill.solid(); card.fill.fore_color.rgb = CARD_WHITE
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Pt(1.5)
        
        # Header banner inside card
        hdr = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, Inches(0.42))
        hdr.fill.solid(); hdr.fill.fore_color.rgb = accent_color
        hdr.line.fill.background()
        tf_h = hdr.text_frame
        p_h = tf_h.paragraphs[0]
        p_h.text = header_title
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = CARD_WHITE
        p_h.alignment = PP_ALIGN.CENTER

        # Content text box
        c_box = slide.shapes.add_textbox(x + Inches(0.12), y + Inches(0.45), w - Inches(0.24), h - Inches(0.5))
        tf_c = c_box.text_frame
        tf_c.word_wrap = True
        return tf_c

    # ─── BOX 1 (Top-Left): OBJECTIVES ────────────────────────────────────────
    tf_obj = make_card(x1, row1_y, col_w, row1_h, "Objectives", BIT_GREEN)
    obj_bullets = [
        "Synthetic Signal Synthesis: Generate raw complex IQ datasets for 4 modulation types (AM, FM, BPSK, QPSK) across -20 dB to +20 dB SNR with AWGN.",
        "Time-Frequency Conversion: Implement STFT to convert 1D time signals into normalized 2D Spectrogram images (64x64).",
        "Deep CNN Architecture: Train a 4-block PyTorch CNN for modulation classification and Occupied/Vacant state detection.",
        "Cooperative Decision Fusion: Build a 3-Node Fusion Center executing OR and Majority Voting rules to overcome fading blind spots.",
        "Hardware-In-The-Loop: Validate over-the-air signal detection using physical RTL-SDR receiver hardware."
    ]
    for i, b in enumerate(obj_bullets):
        p = tf_obj.paragraphs[0] if i == 0 else tf_obj.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(3)

    # ─── BOX 2 (Top-Middle): INTRODUCTION ────────────────────────────────────
    tf_intro = make_card(x2, row1_y, col_w, row1_h, "Introduction", BIT_GREEN)
    intro_bullets = [
        "Spectrum Paradox: Over 70% of licensed radio spectrum sits temporarily idle (White Spaces), despite severe artificial spectrum scarcity.",
        "Cognitive Radio Networks (CRN): Enables unlicensed Secondary Users (SUs) to opportunistically access vacant channels without causing interference.",
        "Flaw of Classical Energy Detectors: Vulnerable to noise uncertainty at low SNR (< -5 dB) and cannot identify signal modulation types.",
        "Hidden Node Problem: Multipath fading and physical shadowing can block a single sensor, causing transmission collisions with Primary Users (PUs).",
        "Our Solution: Combining Deep Learning (CNN) on 2D Spectrograms with 3-Node Cooperative Spatial Diversity (Fusion Center)."
    ]
    for i, b in enumerate(intro_bullets):
        p = tf_intro.paragraphs[0] if i == 0 else tf_intro.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(3)

    # ─── BOX 3 (Top-Right): METHODOLOGY ──────────────────────────────────────
    tf_meth = make_card(x3, row1_y, col_w, row1_h, "Methodology & Architecture", BIT_GREEN)
    meth_bullets = [
        "1. IQ Signal Generation: SciPy synthesis of DSB-FC AM, wideband FM (75 kHz dev), and RRC-filtered BPSK/QPSK (roll-off 0.35).",
        "2. STFT Processing: Windowed FFT (Hann, N=64, overlap=48) -> Power Spectral Density (PSD) -> Min-Max scaled to [0, 1].",
        "3. PyTorch CNN Model: 4 Conv2D blocks (32->64->128->256) + BatchNorm + MaxPool -> GlobalAvgPool -> Dense(128) -> Softmax (422K params).",
        "4. Fusion Center Logic: Aggregates 3 node reports via:",
        "   - OR Rule: Q_d = 1 - (1 - P_d)^3 (Maximizes PU Protection)",
        "   - Majority Rule: Q_d = 3*P_d^2*(1 - P_d) + P_d^3 (Balanced)",
        "5. Hardware Integration: pyrtlsdr reads live 1024 IQ samples from RTL2832U USB stick for real-time over-the-air classification."
    ]
    for i, b in enumerate(meth_bullets):
        p = tf_meth.paragraphs[0] if i == 0 else tf_meth.add_paragraph()
        p.text = "• " + b if not b.startswith("   -") else b
        p.font.size = Pt(9.2)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(2)

    # ─── BOX 4 (Bottom-Left): FIGURES & VISUALS ──────────────────────────────
    tf_fig = make_card(x1, row2_y, col_w, row2_h, "Figures & Visual Analytics", BIT_GREEN)
    
    # Embed real ROC and Accuracy images
    roc_img = os.path.join(RESULTS_DIR, "roc_cooperative.png")
    acc_img = os.path.join(RESULTS_DIR, "accuracy_vs_snr.png")

    if os.path.exists(roc_img) and os.path.exists(acc_img):
        # Place 2 images side-by-side inside the Figures card
        slide.shapes.add_picture(roc_img, x1 + Inches(0.15), row2_y + Inches(0.55), width=Inches(2.25))
        slide.shapes.add_picture(acc_img, x1 + Inches(2.5), row2_y + Inches(0.55), width=Inches(2.25))

        cap_box = slide.shapes.add_textbox(x1 + Inches(0.1), row2_y + Inches(2.25), col_w - Inches(0.2), Inches(1.1))
        tf_cap = cap_box.text_frame
        tf_cap.word_wrap = True
        p_c1 = tf_cap.paragraphs[0]
        p_c1.text = "Fig 1 (Left): ROC Curves (Pd vs Pf) proving Cooperative Fusion dramatically outperforms single-node sensing at low SNR (-10 dB)."
        p_c1.font.size = Pt(8.5); p_c1.font.color.rgb = TEXT_MUTED

        p_c2 = tf_cap.add_paragraph()
        p_c2.text = "Fig 2 (Right): Classification Accuracy vs SNR demonstrating >95% accuracy for AM/FM and robust performance across -20 dB to +20 dB."
        p_c2.font.size = Pt(8.5); p_c2.font.color.rgb = TEXT_MUTED; p_c2.space_before = Pt(2)

    # ─── BOX 5 (Bottom-Middle): RESULTS ──────────────────────────────────────
    tf_res = make_card(x2, row2_y, col_w, row2_h, "Results & Experimental Validation", BIT_GREEN)
    res_bullets = [
        "Overall Model Accuracy: 71.0% across extreme SNR sweep (-20 dB to +20 dB); exceeds 95% at SNR > 0 dB.",
        "Per-Class Performance: AM (F1 = 0.95), FM (F1 = 0.95), BPSK (F1 = 0.62, identified as digital carrier).",
        "Cooperative Sensing Gain: At -10 dB SNR, individual node Pd is ~50%, while OR-rule fusion boosts Pd to > 85%.",
        "Live Hardware Validation (Fitipower FC0012 RTL-SDR):",
        "   - 91.5 / 93.5 / 104.0 MHz: Detected FM Primary Users (OCCUPIED, 92%)",
        "   - 98.3 MHz: Discovered Spectrum Hole / White Space (VACANT, 49.1%)",
        "Phase 2 Enhancements: Autonomous channel hopping, soft fusion combining, and SSDF malicious node trust defense."
    ]
    for i, b in enumerate(res_bullets):
        p = tf_res.paragraphs[0] if i == 0 else tf_res.add_paragraph()
        p.text = "• " + b if not b.startswith("   -") else b
        p.font.size = Pt(9.2)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(2.5)

    # ─── BOX 6 (Bottom-Right): ACKNOWLEDGEMENT & CONCLUSION ──────────────────
    tf_ack = make_card(x3, row2_y, col_w, row2_h, "Conclusion & Acknowledgement", BIT_GREEN)
    ack_bullets = [
        "Conclusion: Successfully demonstrated that Deep Learning (CNN) + Cooperative Fusion effectively detects Primary Users and harvests White Spaces in real time.",
        "Future Scope: Full-duplex secondary transmission via HackRF/USRP, Deep Q-Learning (DRL) for predictive frequency hopping, and soft Likelihood Ratio fusion.",
        "Acknowledgement: We express our sincere gratitude to Bangalore Institute of Technology (BIT) for laboratory resources, our guide Prof. Asha R, and the Department of Electronics & Telecommunication Engineering for their invaluable support.",
        "References: [1] J. Mitola, 'Cognitive Radio: An Integrated Agent Architecture', IEEE Comm., 1999.  [2] T. O'Shea et al., 'Convolutional Radio Modulation Recognition', IEEE Trans., 2018."
    ]
    for i, b in enumerate(ack_bullets):
        p = tf_ack.paragraphs[0] if i == 0 else tf_ack.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(9.0)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(3)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Footer: Additional Information / Contact Box
    # ─────────────────────────────────────────────────────────────────────────
    footer_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(10.55), Inches(15.335), Inches(0.55)
    )
    footer_box.fill.solid(); footer_box.fill.fore_color.rgb = CARD_WHITE
    footer_box.line.color.rgb = BORDER_GRAY
    footer_box.line.width = Pt(1.5)
    tf_ftr = footer_box.text_frame
    tf_ftr.word_wrap = True
    p_ftr = tf_ftr.paragraphs[0]
    p_ftr.text = "Additional Information / Contact: Dept. of Electronics and Telecommunication Engineering, Bangalore Institute of Technology, K.R. Road, V.V. Puram, Bengaluru – 560004 | Email: project.g10.ete@bit-bangalore.edu.in | Web Dashboard: http://localhost:8501"
    p_ftr.font.size = Pt(9.5)
    p_ftr.font.bold = True
    p_ftr.font.color.rgb = BIT_GREEN
    p_ftr.alignment = PP_ALIGN.CENTER

    prs.save(OUTPUT_PPTX)
    print(f"[OK] A3 Presentation Poster generated: {OUTPUT_PPTX}")
    return OUTPUT_PPTX


if __name__ == "__main__":
    create_poster()
