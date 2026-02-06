import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv

# --- STABLE IMPORTS ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents import create_csv_agent
    from langchain_community.callbacks import get_openai_callback  
    import tabulate 
except ImportError:
    st.error("Missing libraries! Run: pip install langchain-openai langchain-experimental python-dotenv tabulate langchain-community")

# --- LOAD ENVIRONMENT ---
load_dotenv()
OS_API_KEY = os.getenv("OPENAI_API_KEY")

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="AI ICU Patient Monitor", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.88)), 
        url("https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&q=80&w=2070");
        background-size: cover;
    }}
    @keyframes blinker {{ 50% {{ opacity: 0; }} }}
    .emergency-alert {{
        color: white; background-color: #B71C1C; padding: 25px;
        text-align: center; font-weight: bold; font-size: 30px;
        border-radius: 12px; animation: blinker 1s linear infinite;
        border: 5px solid #FF5252; margin-bottom: 25px;
        box-shadow: 0px 0px 20px rgba(255,0,0,0.5);
    }}
    .telemetry-footer {{
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #0D0D0D; color: #00FF41;
        text-align: center; padding: 15px;
        font-family: 'Courier New', monospace; font-size: 15px;
        font-weight: bold; border-top: 2px solid #333; z-index: 9999;
    }}
    </style>
""", unsafe_allow_html=True)

if 'telemetry' not in st.session_state:
    st.session_state.telemetry = "SYSTEM ONLINE >> Model: GPT-4o | Latency: 0.0s | Total Tokens: 0"

# --- DATA GENERATOR ---
def generate_patient_files():
    minutes = np.arange(0, 60)
    
    # 1. Patient: Sepsis (High Temp, Dropping BP)
    p1 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': np.random.randint(90, 110, 60) + np.linspace(0, 30, 60).astype(int),
        'Temperature': np.round(np.linspace(98.6, 104.2, 60) + np.random.normal(0, 0.2, 60), 1),
        'BP_Systolic': np.linspace(120, 82, 60).astype(int),
        'Blood_Oxygen': np.linspace(98, 89, 60).astype(int)
    })
    
    # 2. Patient: Arrhythmia (Sudden V-Tach Spike)
    hb_v_tach = [75 + np.random.randint(-5, 5) for _ in range(40)] + [180 + np.random.randint(5, 15) for _ in range(20)]
    p2 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': hb_v_tach,
        'Temperature': np.round(98.6 + np.random.normal(0, 0.1, 60), 1),
        'BP_Systolic': [120 + np.random.randint(-5, 5) for _ in range(40)] + [85 + np.random.randint(-10, 5) for _ in range(20)],
        'Blood_Oxygen': [98] * 40 + [88] * 20
    })
    
    # 3. Patient: Respiratory Failure (Progressive SpO2 Drop)
    p3 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': np.linspace(70, 120, 60).astype(int),
        'Temperature': np.round(98.6 + np.random.normal(0, 0.1, 60), 1),
        'BP_Systolic': np.linspace(115, 95, 60).astype(int),
        'Blood_Oxygen': np.linspace(96, 78, 60).astype(int)
    })
    
    p1.to_csv('patient_1_sepsis.csv', index=False)
    p2.to_csv('patient_2_arrhythmia.csv', index=False)
    p3.to_csv('patient_3_respiratory.csv', index=False)

# --- AGENTIC RAG ENGINE ---
def run_clinical_agent(file_path, query):
    if not OS_API_KEY: return "⚠️ API Key missing."
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OS_API_KEY)
    agent = create_csv_agent(llm, file_path, verbose=False, agent_type="openai-functions", allow_dangerous_code=True)
    start_t = time.time()
    with get_openai_callback() as cb:
        try:
            response = agent.run(query)
            tokens = cb.total_tokens
        except Exception as e:
            response = f"Error: {str(e)}"; tokens = 0
    latency = round(time.time() - start_t, 2)
    st.session_state.telemetry = f"TELEMETRY LOG >> Model: GPT-4o | Latency: {latency}s | Tokens: {tokens}"
    return response

# --- MAIN APP ---
def main():
    if not os.path.exists('patient_1_sepsis.csv'): generate_patient_files()

    st.sidebar.title("🏥 ICU Control Panel")
    
    # Checkbox for audio interaction (Browser Requirement)
    sound_on = st.sidebar.checkbox("🔊 Enable Alarm Audio", value=True)
    
    patient_file = st.sidebar.selectbox("Active Patient Stream", 
                                       ["patient_1_sepsis.csv", "patient_2_arrhythmia.csv", "patient_3_respiratory.csv"])

    df = pd.read_csv(patient_file)
    last_row = df.iloc[-1]

    st.title("AI-Based Patient Monitor")
    st.markdown(f"**Monitoring Stream:** `{patient_file}`")
    st.markdown("---")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Heart Rate", f"{last_row['Heartbeat']} BPM")
    m2.metric("Temperature", f"{last_row['Temperature']} °F")
    m3.metric("BP Systolic", f"{last_row['BP_Systolic']} mmHg")
    m4.metric("SpO2", f"{last_row['Blood_Oxygen']}%")

    # Alarm Logic (Critical thresholds)
    is_critical = (last_row['Blood_Oxygen'] < 90 or last_row['Heartbeat'] > 155 or last_row['Temperature'] > 103)
    
    if is_critical:
        st.markdown('<div class="emergency-alert">🚨 CRITICAL PATIENT ALARM: IMMEDIATE ATTENTION 🚨</div>', unsafe_allow_html=True)
        
        if sound_on:
            # Professional medical triple-beep
            alarm_url = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
            st.markdown(f"""
                <audio autoplay loop>
                    <source src="{alarm_url}" type="audio/ogg">
                </audio>
                """, unsafe_allow_html=True)
        
        with st.expander("🤖 AGENTIC CLINICAL RECOMMENDATION", expanded=True):
            prompt = "Analyze the historical vitals. Diagnose the condition and suggest 3 life-saving interventions."
            with st.spinner("Agent analyzing clinical history..."):
                answer = run_clinical_agent(patient_file, prompt)
                st.write(answer)

    # Trends & RAG
    st.subheader("60-Minute Vital Sign Trends")
    st.line_chart(df.set_index('Timestamp')[['Heartbeat', 'Blood_Oxygen']])
    
    st.divider()
    user_q = st.text_input("🔍 Query Clinical Record (RAG):")
    if user_q:
        st.info(run_clinical_agent(patient_file, user_q))

    st.markdown(f'<div class="telemetry-footer">{st.session_state.telemetry}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()