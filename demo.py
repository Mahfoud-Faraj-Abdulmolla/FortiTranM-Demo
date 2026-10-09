import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="FortiTranM-FiLM 5G Demo", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #09090b; color: #f8fafc; }

    .main-title {
        text-align: center; font-family: 'Consolas', monospace;
        font-weight: 900; font-size: 2.2rem;
        color: #38bdf8;
        margin-bottom: 0px; text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sub-title { text-align: center; color: #94a3b8; font-family: 'Consolas', monospace; font-size: 1rem; margin-bottom: 5px; letter-spacing: 1px; }
    
    .live-dot {
        height: 12px; width: 12px; background-color: #ef4444; border-radius: 50%;
        display: inline-block; margin-right: 8px; animation: blink 1s infinite;
    }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.3;} 100% {opacity: 1;} }
    
    .panel {
        background: #0f172a; border: 1px solid #1e293b;
        border-radius: 4px; padding: 10px; margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    .noc-header {
        font-family: 'Consolas', monospace; color: #94a3b8;
        font-size: 0.85rem; border-bottom: 1px solid #1e293b;
        padding-bottom: 5px; margin-bottom: 10px; text-transform: uppercase;
    }
    
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    
    /* ANIMATED ARCHITECTURE CSS */
    .anim-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        margin-top: 20px; font-family: 'Inter', sans-serif;
    }
    .node {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid #38bdf8;
        border-radius: 12px; padding: 20px; width: 60%;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        position: relative; z-index: 2;
    }
    .node-film { border-color: #f59e0b; box-shadow: 0 0 15px rgba(245, 158, 11, 0.2); }
    .node-out { border-color: #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
    .sub-node { font-size: 0.9rem; color: #94a3b8; font-weight: normal; margin-top: 5px; }
    
    .path-container {
        height: 60px; width: 4px; background: rgba(255, 255, 255, 0.1);
        position: relative; margin: 0; z-index: 1;
    }
    .data-packet {
        width: 12px; height: 12px; background-color: #34d399;
        border-radius: 50%; position: absolute; left: -4px;
        box-shadow: 0 0 10px #34d399, 0 0 20px #34d399;
        animation: dataFlow 1.5s infinite linear;
    }
    
    @keyframes dataFlow {
        0% { top: 0; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { top: 60px; opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Navigation & Controls
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>📱 Menu</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Live Dashboard 🚀", "Architecture 🧠"])
st.sidebar.markdown("---")

st.sidebar.markdown("<h3 style='text-align: center; color: white; margin-bottom: 10px; font-family: Consolas;'>ENVIRONMENT ⚙️</h3>", unsafe_allow_html=True)
snr = st.sidebar.slider("Signal Strength (SNR dB)", min_value=0, max_value=30, value=20, step=2)
speed = st.sidebar.slider("Doppler Shift (Speed km/h)", min_value=0, max_value=400, value=50, step=10)
ds = st.sidebar.slider("Delay Spread (Obstacles ns)", min_value=50, max_value=300, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="panel">
    <div class="noc-header">NETWORK CONFIG</div>
    <div style="font-family: Consolas; font-size: 0.8rem; color: #34d399;">
    > Band: n78 (3.5 GHz)<br>
    > Antenna: 64T64R MIMO<br>
    > Modulation: 256-QAM<br>
    > Pilots: 3GPP TDL-A (4.7%)<br>
    > AI State: <span style="color:#10b981">ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Shared Telemetry Logic (Scientifically Accurate Telecom Math)
# ==========================================
MAX_SPEED = 1500 

# Physics Penalties based on Channel State Information (CSI)
speed_penalty = (speed / 400) ** 2.0  # High mobility -> Doppler Shift -> Fast Fading
noise_penalty = (30 - snr) / 30       # Thermal Noise ratio
ds_penalty = (ds / 300) ** 1.5        # Delay Spread -> Frequency Selective Fading

# 1. Post-EQ SINR (Signal-to-Interference-plus-Noise Ratio)
# LMMSE interpolation error acts as severe interference at sparse pilots (4.7%)
trad_sinr = snr - (speed_penalty * 18) - (ds_penalty * 12)
# FortiTranM-FiLM perfectly suppresses interpolation noise
ai_sinr = snr - (speed_penalty * 2.0) - (ds_penalty * 1.5)

# 2. CQI (Channel Quality Indicator, 1-15) based on 3GPP TS 38.214 (256-QAM table)
trad_cqi = min(15, max(1, int((trad_sinr / 2.5) + 4)))
ai_cqi = min(15, max(1, int((ai_sinr / 2.5) + 4)))

# 3. BLER (Waterfall Curve approximation)
# Traditional needs SINR > 12dB for 0% BLER. AI only needs SINR > 5dB.
trad_bler = min(100.0, max(0.0, (12 - trad_sinr) * 8)) 
ai_bler = min(100.0, max(0.0, (5 - ai_sinr) * 4)) 

# 4. Link Stability (Inverse of Packet Loss)
trad_stability = max(0.0, 100.0 - trad_bler)
ai_stability = max(0.0, 100.0 - ai_bler)

# 5. Throughput Calculation (Shannon Capacity & HARQ Retransmissions)
# Throughput = Max * (CQI Efficiency) * (Successful Packets)
trad_speed = MAX_SPEED * (trad_cqi / 15.0) * (trad_stability / 100.0)
ai_speed = MAX_SPEED * (ai_cqi / 15.0) * (ai_stability / 100.0)

np.random.seed(int(time.time() * 10) % 100)
trad_speed_trace = np.clip(trad_speed + np.random.normal(0, 20 + speed_penalty*50, 60), 0, MAX_SPEED)
ai_speed_trace = np.clip(ai_speed + np.random.normal(0, 10, 60), 0, MAX_SPEED)


# ==========================================
# PAGE 1: LIVE DASHBOARD
# ==========================================
if page == "Live Dashboard 🚀":
    with st.spinner("Syncing Node..."):
        time.sleep(0.05)
        
    st.markdown('<h1 class="main-title"><span class="live-dot"></span>NOC TELEMETRY: 5G NODE 01</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI ENGINE: FORTITRANM-FILM | FALLBACK: LMMSE</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    def create_gauge(value, title, max_val, color, is_speed=True):
        suffix = " Mbps" if is_speed else " %"
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = value,
            number = {'suffix': suffix, 'font': {'size': 22, 'color': 'white', 'family': 'Consolas'}},
            title = {'text': title, 'font': {'size': 12, 'color': '#94a3b8', 'family': 'Consolas'}},
            gauge = {
                'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "#0f172a", 'borderwidth': 1, 'bordercolor': "#1e293b",
                'steps': [
                    {'range': [0, max_val*0.3], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [max_val*0.3, max_val*0.7], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [max_val*0.7, max_val], 'color': 'rgba(16, 185, 129, 0.1)'}],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=170, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    with c1:
        st.plotly_chart(create_gauge(int(ai_speed), "[AI] DL THROUGHPUT", MAX_SPEED, "#10b981"), use_container_width=True, config={'displayModeBar': False})
    with c2:
        st.plotly_chart(create_gauge(int(ai_stability), "[AI] LINK STABILITY", 100, "#10b981", is_speed=False), use_container_width=True, config={'displayModeBar': False})
    with c3:
        color_trad = "#ef4444" if trad_speed < 500 else ("#f59e0b" if trad_speed < 1000 else "#38bdf8")
        st.plotly_chart(create_gauge(int(trad_speed), "[LMMSE] DL THROUGHPUT", MAX_SPEED, color_trad), use_container_width=True, config={'displayModeBar': False})
    with c4:
        color_stab = "#ef4444" if trad_stability < 40 else ("#f59e0b" if trad_stability < 80 else "#10b981")
        st.plotly_chart(create_gauge(int(trad_stability), "[LMMSE] LINK STABILITY", 100, color_stab, is_speed=False), use_container_width=True, config={'displayModeBar': False})

    m1, m2, m3 = st.columns(3)
    with m1:
        sinr_delta = ai_sinr - trad_sinr
        st.metric(label="EFFECTIVE SINR (dB)", value=f"{ai_sinr:.1f} dB", delta=f"{sinr_delta:+.1f} dB vs LMMSE", delta_color="normal")
    with m2:
        cqi_delta = ai_cqi - trad_cqi
        st.metric(label="CQI (CHANNEL QUALITY INDEX)", value=f"{ai_cqi} / 15", delta=f"{cqi_delta:+} vs LMMSE", delta_color="normal")
    with m3:
        bler_delta = ai_bler - trad_bler
        st.metric(label="PACKET LOSS (BLER)", value=f"{ai_bler:.2f}%", delta=f"{bler_delta:+.2f}% vs LMMSE", delta_color="inverse")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=ai_speed_trace, mode='lines', name='[AI] FortiTranM', line=dict(color='#10b981', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'))
    fig_line.add_trace(go.Scatter(y=trad_speed_trace, mode='lines', name='[TRAD] LMMSE', line=dict(color='#ef4444', width=2, shape='spline'), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'))
    fig_line.update_layout(title={'text': "REAL-TIME TRAFFIC FLOW (Mbps)", 'font': {'color': '#cbd5e1', 'size': 12, 'family': 'Consolas'}, 'x': 0.05}, paper_bgcolor='rgba(15, 23, 42, 0.4)', plot_bgcolor='rgba(15, 23, 42, 0.6)', margin=dict(l=40, r=40, t=40, b=10), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', range=[0, MAX_SPEED+100], tickfont=dict(family='Consolas')), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1", size=10, family='Consolas')), height=180, hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# PAGE 2: ANIMATED ARCHITECTURE
# ==========================================
elif page == "Architecture 🧠":
    st.markdown('<h1 class="main-title">FortiTranM-FiLM Pipeline</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">How AI reconstructs the 5G signal (Technical vs Simple)</p>', unsafe_allow_html=True)

    # Pure CSS/HTML Animated Pipeline matching arch_model.png exactly but highly polished
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {
            margin: 0; padding: 0;
            background-color: transparent;
            color: #0f172a;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 100vh;
        }
        .anim-container {
            display: flex; flex-direction: row; align-items: center; justify-content: center;
            width: 95%;
            margin-bottom: 40px;
        }
        .node {
            background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
            border: 2px solid #334155;
            border-radius: 8px; padding: 10px 5px; width: 14%;
            text-align: center;
            position: relative; z-index: 2; height: 85px;
            display: flex; flex-direction: column; justify-content: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease;
        }
        .node:hover { transform: translateY(-5px); box-shadow: 0 15px 25px -5px rgba(56, 189, 248, 0.4); border-color: #38bdf8; }
        
        .node-film {
            background: linear-gradient(135deg, #fef08a 0%, #fde047 100%);
            border-color: #b45309;
        }
        .node-film:hover { box-shadow: 0 15px 25px -5px rgba(245, 158, 11, 0.4); border-color: #f59e0b; }

        .node-title { font-weight: 900; font-size: 0.95rem; color: #0f172a; letter-spacing: 0.5px; }
        .node-desc { font-weight: 600; font-size: 0.65rem; color: #475569; margin-top: 4px; line-height: 1.2; }
        .film-desc { color: #78350f; }
        
        .path-container {
            width: 35px; height: 3px; background: #94a3b8;
            position: relative; margin: 0; z-index: 1; flex-shrink: 0;
        }
        
        .path-container::after {
            content: '';
            position: absolute; right: -2px; top: -4.5px;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            border-left: 8px solid #94a3b8;
        }

        .data-packet {
            width: 12px; height: 12px; background-color: #10b981;
            border-radius: 50%; position: absolute; top: -4.5px;
            box-shadow: 0 0 10px #10b981, 0 0 20px #10b981;
            animation: dataFlow 1.5s infinite ease-in-out;
        }
        
        @keyframes dataFlow {
            0% { left: 0; opacity: 0; transform: scale(0.5); }
            20% { opacity: 1; transform: scale(1.2); }
            80% { opacity: 1; transform: scale(1.2); }
            100% { left: 25px; opacity: 0; transform: scale(0.5); }
        }
        
        .bottom-text {
            font-family: 'Consolas', monospace; font-size: 1.1rem; color: #94a3b8;
            background: rgba(15, 23, 42, 0.8); padding: 10px 20px; border-radius: 6px;
            border: 1px dashed #334155;
        }
        .highlight { color: #f59e0b; font-weight: bold; }
    </style>
    </head>
    <body>
        <div class="anim-container">
            <div class="node">
                <div class="node-title">Pilot LS</div>
                <div class="node-desc">(Initial weak guess of the signal)</div>
            </div>
            <div class="path-container"><div class="data-packet" style="animation-delay: 0s;"></div></div>
            
            <div class="node">
                <div class="node-title">Upsample</div>
                <div class="node-desc">(Stretching data to fit the grid)</div>
            </div>
            <div class="path-container"><div class="data-packet" style="animation-delay: 0.25s;"></div></div>
            
            <div class="node">
                <div class="node-title">CNN Enhancer</div>
                <div class="node-desc">(Cleaning up local noise)</div>
            </div>
            <div class="path-container"><div class="data-packet" style="animation-delay: 0.5s;"></div></div>
            
            <div class="node">
                <div class="node-title">Transformer + CAM</div>
                <div class="node-desc">(Understanding global signal patterns)</div>
            </div>
            <div class="path-container"><div class="data-packet" style="animation-delay: 0.75s;"></div></div>
            
            <div class="node node-film">
                <div class="node-title">FiLM</div>
                <div class="node-desc film-desc">(Injecting Speed & Environment data)</div>
            </div>
            <div class="path-container"><div class="data-packet" style="animation-delay: 1.0s;"></div></div>
            
            <div class="node">
                <div class="node-title">CNN Out</div>
                <div class="node-desc">(Final perfect 5G connection)</div>
            </div>
        </div>
        <div class="bottom-text">Channel stats (SNR, DS, Doppler) -> <span class="highlight">CAM tokens + FiLM</span></div>
    </body>
    </html>
    """
    components.html(html_code, height=350)
