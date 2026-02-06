import pandas as pd
import numpy as np

def generate_patient_data(filename, condition):
    minutes = np.arange(0, 60)
    data = {'Timestamp': [f"12:{i:02d} PM" for i in minutes]}
    
    if condition == 'sepsis':
        # Increasing temperature, decreasing BP and Oxygen
        data['Heartbeat'] = np.random.randint(90, 110, 60) + np.linspace(0, 30, 60).astype(int)
        data['Temperature'] = np.round(np.linspace(98.6, 103.5, 60) + np.random.normal(0, 0.2, 60), 1)
        data['BP_Systolic'] = np.linspace(120, 85, 60).astype(int)
        data['Blood_Oxygen'] = np.linspace(98, 91, 60).astype(int)
    
    elif condition == 'arrhythmia':
        # Normal until minute 40, then spike in heartbeat (V-Tach)
        hb = [75 + np.random.randint(-5, 5) for _ in range(40)] + [160 + np.random.randint(10, 30) for _ in range(20)]
        data['Heartbeat'] = hb
        data['Temperature'] = np.round(98.6 + np.random.normal(0, 0.1, 60), 1)
        data['BP_Systolic'] = [120 + np.random.randint(-5, 5) for _ in range(40)] + [90 + np.random.randint(-10, 0) for _ in range(20)]
        data['Blood_Oxygen'] = [97] * 40 + [92] * 20
        
    elif condition == 'respiratory_failure':
        # Progressive decline in Oxygen
        data['Heartbeat'] = np.linspace(70, 110, 60).astype(int)
        data['Temperature'] = np.round(98.6 + np.random.normal(0, 0.1, 60), 1)
        data['BP_Systolic'] = np.linspace(120, 100, 60).astype(int)
        data['Blood_Oxygen'] = np.linspace(96, 82, 60).astype(int)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated: {filename}")

# Generate the three files
generate_patient_data('patient_1_sepsis.csv', 'sepsis')
generate_patient_data('patient_2_arrhythmia.csv', 'arrhythmia')
generate_patient_data('patient_3_respiratory.csv', 'respiratory_failure')