import librosa
import numpy as np

def get_health_scores(file):
    y, sr = librosa.load(file, sr=16000)
    
    #ค่าเฉลี่ย Spectral Centroid
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    #ค่าความเงียบ (RMS)
    rms = librosa.feature.rms(y=y)[0]
    #ค่าความแปรปรวน (Onset Strength)
    os = np.std(librosa.onset.onset_strength(y=y, sr=sr))
    #OngsaTestGithub
    # คำนวณเป็น % Logic
    feno = min((sc / 5000) * 100, 100)
    odi = min((np.sum(rms < 0.01) / len(rms)) * 500, 100)
    stress = min(os * 15, 100)
    
    return feno, odi, stress, y, sr

def calculate_reliability(y):
    # คำนวณหาพลังงานเสียงเพื่อเช็คความชัดเจน
    signal_power = np.mean(y**2)
    # สุ่มเช็ค Noise จากช่วงต้นไฟล์ (5% แรก)
    noise_power = np.var(y[:int(len(y)*0.05)]) 
    
    # คำนวณ SNR (Signal-to-Noise Ratio)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    # แปลงเป็น % (สมมติ 20dB คือชัด 100%)
    reliability = min(max((snr / 20) * 100, 0), 100)
    return reliability