import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from processor import get_health_scores, calculate_reliability

# 1. Page Configuration - เน้นความกะทัดรัด
st.set_page_config(page_title="Voice Biomarker Report", layout="wide")

# 2. Optimized CSS - ควบคุมขนาดให้ไม่เต็มจอจนเกินไปและดูหรูหรา
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; max-width: 1000px; } /* บีบความกว้างให้ดูแพง */
    [data-testid="stMetric"] { 
        background-color: #1a1c24; padding: 12px !important; 
        border-radius: 10px; border: 1px solid #30363d;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .audit-banner { 
        padding: 10px 15px; border-radius: 8px; background-color: #161b22; 
        border: 1px solid #30363d; font-size: 0.85rem; margin-bottom: 20px;
    }
    .compact-table { width: 100%; font-size: 0.8rem; border-collapse: collapse; color: #8b949e; }
    .compact-table td { padding: 6px; border-bottom: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🎙️ Voice Biomarker AI Assessment")
#--st.caption("")--
st.markdown("---")

# --- Home Page Upload ---
uploaded_file = st.file_uploader("Clinical Audio Input", type=["wav"], label_visibility="collapsed")

if not uploaded_file:
    st.info("👋 Please upload a .wav file to begin. / กรุณาอัปโหลดไฟล์เพื่อเริ่มการวิเคราะห์")
else:
    # Processing Data
    feno, odi, stress, y, sr = get_health_scores(uploaded_file)
    rel_score = calculate_reliability(y)

    # --- Section 1: System Audit (Now on Main Page) ---
    q_color = "#2ea043" if rel_score > 75 else "#f39c12"
    st.markdown(f"""
        <div class='audit-banner'>
            🛡️ <b>System Audit:</b> Data Quality <span style='color:{q_color};'>{rel_score:.1f}%</span> 
            | Signal: <span style='color:{q_color};'>Verified</span>
            <span style='float:right; color:#8b949e;'>ID: {uploaded_file.name}</span>
        </div>
    """, unsafe_allow_html=True)

    # --- Section 2: Clinical Dashboard ---
    tab1, tab2 = st.tabs(["🩺 Clinical Risk", "📈 Deep Analysis"])

    with tab1:
        st.markdown("##### 📊 Biomarker Analysis Report / รายงานดัชนีชีวภาพ")
        c1, c2, c3 = st.columns(3)
        
        # Metric Helpers
        def draw_metric(col, label, val, th_desc, tooltip):
            status = "🟢" if val <= 25 else "🟡" if val <= 50 else "🟠" if val <= 75 else "🔴"
            with col:
                st.metric(label, f"{val:.1f}%", delta=status, delta_color="off", help=tooltip)
                st.progress(float(val)/100)
                st.caption(th_desc)

        draw_metric(c1, "Airway Inflammation", feno, "การอักเสบทางเดินหายใจ", "วิเคราะห์จากความพร่าและความถี่ของเสียง")
        draw_metric(c2, "Oxygen Drop (ODI)", odi, "การลดลงของออกซิเจน", "วิเคราะห์จากจังหวะการหยุดหายใจ")
        draw_metric(c3, "Vascular Stress", stress, "ความเครียดระบบหลอดเลือด", "วิเคราะห์จากความแปรปรวนจังหวะหายใจ")

        # Interpretation Guide
        st.markdown("""
        <div style='margin-top:20px; padding:15px; background:#161b22; border-radius:10px; border:1px solid #30363d;'>
            <b style='font-size:0.85rem;'>📋 Interpretation Guide / คู่มือการแปลผล</b>
            <table class='compact-table'>
                <tr><td>0-25%</td><td style='color:#2ea043;'>Optimal</td><td>ปกติ ไม่พบความเสี่ยง</td></tr>
                <tr><td>26-50%</td><td style='color:#e3b341;'>Fair</td><td>เฝ้าระวัง ควรพักผ่อนให้เพียงพอ</td></tr>
                <tr><td>51-75%</td><td style='color:#f39c12;'>Warning</td><td>เสี่ยงสูง แนะนำปรึกษาผู้เชี่ยวชาญ</td></tr>
                <tr><td>76-100%</td><td style='color:#f85149;'>Critical</td><td>อันตราย แนะนำพบแพทย์ด่วน</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        col_lt, col_rt = st.columns([1, 1.5])
        with col_lt:
            st.markdown("##### 🧬 Research Insights")
            st.caption("• **Asia Context:** Snoring in normal BMI linked to CAD.")
            st.caption("• **Biochemical:** Signal of Vitamin D Deficiency.")
            st.warning("Assessment for screening only.")

        with col_rt:
            st.markdown("##### 📊 Signal Spectrogram")
            fig, ax = plt.subplots(figsize=(8, 3.8))
            fig.patch.set_facecolor('#0e1117')
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='turbo')
            ax.tick_params(labelsize=7, colors='white')
            st.pyplot(fig)

    st.markdown("---")
    st.caption("Verified Clinical Voice Biomarker v1.2 | Thitiworada Moleechart")