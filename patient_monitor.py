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
except ImportError:
    st.error("Missing libraries! Run: pip install langchain-openai langchain-experimental python-dotenv")

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
OS_API_KEY = os.getenv("OPENAI_API_KEY")

# --- CONFIGURATION ---
st.set_page_config(page_title="AI ICU Patient Monitor", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
    @keyframes blinker { 50% { opacity: 0; } }
    .emergency-alert {
        color: white; background-color: #FF0000; padding: 20px;
        text-align: center; font-weight: bold; font-size: 28px;
        border-radius: 10px; animation: blinker 1s linear infinite;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA GENERATOR ---
def generate_patient_files():
    minutes = np.arange(0, 60)
    # Patient 1: Sepsis
    p1 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': np.random.randint(90, 110, 60) + np.linspace(0, 30, 60).astype(int),
        'Temperature': np.round(np.linspace(98.6, 104.2, 60) + np.random.normal(0, 0.2, 60), 1),
        'BP_Systolic': np.linspace(120, 82, 60).astype(int),
        'Blood_Oxygen': np.linspace(98, 89, 60).astype(int)
    })
    # Patient 2: Arrhythmia (V-Tach)
    hb_v_tach = [75 + np.random.randint(-5, 5) for _ in range(40)] + [180 + np.random.randint(5, 15) for _ in range(20)]
    p2 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': hb_v_tach,
        'Temperature': np.round(98.6 + np.random.normal(0, 0.1, 60), 1),
        'BP_Systolic': [120 + np.random.randint(-5, 5) for _ in range(40)] + [85 + np.random.randint(-10, 5) for _ in range(20)],
        'Blood_Oxygen': [98] * 40 + [88] * 20
    })
    # Patient 3: Respiratory Failure
    p3 = pd.DataFrame({
        'Timestamp': [f"12:{i:02d} PM" for i in minutes],
        'Heartbeat': np.linspace(70, 120, 60).astype(int),
        'Temperature': np.round(98.6 + np.random.normal(0, 0.1, 60), 1),
        'BP_Systolic': np.linspace(115, 95, 60).astype(int),
        'Blood_Oxygen': np.linspace(96, 80, 60).astype(int)
    })
    p1.to_csv('patient_1_sepsis.csv', index=False)
    p2.to_csv('patient_2_arrhythmia.csv', index=False)
    p3.to_csv('patient_3_respiratory.csv', index=False)

# --- AGENTIC RAG ENGINE ---
def run_clinical_agent(file_path, query):
    if not OS_API_KEY:
        return "⚠️ ERROR: OpenAI API Key not found in .env file.", 0
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OS_API_KEY)
    
    agent = create_csv_agent(
        llm, file_path, verbose=False, 
        agent_type="openai-functions", 
        allow_dangerous_code=True
    )
    
    start_t = time.time()
    try:
        response = agent.run(query)
    except Exception as e:
        response = f"Agent Error: {str(e)}"
    
    latency = round(time.time() - start_t, 2)
    return response, latency

# --- MAIN APP ---
def main():
    if not os.path.exists('patient_1_sepsis.csv'):
        generate_patient_files()

    st.sidebar.title("🏥 ICU Control Panel")
    
    # Check if key is loaded
    if not OS_API_KEY:
        st.sidebar.error("API Key missing! Check .env")
    else:
        st.sidebar.success("API Key Loaded via .env")

    patient_file = st.sidebar.selectbox("Select Patient", 
                                       ["patient_1_sepsis.csv", "patient_2_arrhythmia.csv", "patient_3_respiratory.csv"])

    df = pd.read_csv(patient_file)
    last_row = df.iloc[-1]

    st.title("AI-Based Patient Monitor")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Heart Rate", f"{last_row['Heartbeat']} BPM")
    col2.metric("Temp", f"{last_row['Temperature']} °F")
    col3.metric("BP Systolic", f"{last_row['BP_Systolic']} mmHg")
    col4.metric("SpO2", f"{last_row['Blood_Oxygen']}%")

    # Alarm Logic
    is_emergency = (last_row['Blood_Oxygen'] < 90 or last_row['Heartbeat'] > 150)
    if is_emergency:
        st.markdown('<div class="emergency-alert">🚨 CRITICAL PATIENT ALARM 🚨</div>', unsafe_allow_html=True)
        
        with st.expander("🤖 AGENTIC INTERVENTION PROTOCOL", expanded=True):
            prompt = "Review the patient data trends in the CSV. Diagnose the condition and suggest 3 lifesaving interventions."
            answer, lat = run_clinical_agent(patient_file, prompt)
            st.write(answer)
            st.caption(f"Telemetry: Model: GPT-4o | Latency: {lat}s")

    # Visualization
    st.subheader("60-Minute Vital Trends")
    st.line_chart(df.set_index('Timestamp')[['Heartbeat', 'Blood_Oxygen']])

    # RAG Search
    st.divider()
    st.subheader("🔍 Ask the Patient Record")
    query = st.text_input("Query the agent about this patient's history:")
    if query:
        with st.spinner("Searching records..."):
            ans, _ = run_clinical_agent(patient_file, query)
            st.success(ans)

if __name__ == "__main__":
    main()