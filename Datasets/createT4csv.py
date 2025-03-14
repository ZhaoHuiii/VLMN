import os
import pandas as pd

def process_patient_folder(patient_folder, type):
    data = []

    file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_originalCT.npy")

    data.append({'ct_path': file_path, 'type': type})

    return data

def generate_csv(data_dir1, data_dir2, data_dir3, data_dir4, output_csv, type0, type1, type2, type3):
    all_data = []

    for patient_folder in os.listdir(data_dir1):
        patient_path = os.path.join(data_dir1, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type0)
            all_data.extend(patient_data)

    for patient_folder in os.listdir(data_dir2):
        patient_path = os.path.join(data_dir2, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type1)
            all_data.extend(patient_data)
    
    for patient_folder in os.listdir(data_dir3):
        patient_path = os.path.join(data_dir3, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type2)
            all_data.extend(patient_data)
    
    for patient_folder in os.listdir(data_dir4):
        patient_path = os.path.join(data_dir4, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type3)
            all_data.extend(patient_data)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)

data_dir1 = 'Datasets/processData_T1'
data_dir2 = 'Datasets/processData_T2'
data_dir3 = 'Datasets/processData_T3'
data_dir4 = 'Datasets/processData_T4'
output_csv = 'Datasets/ct4_original.csv'
type0 = 0
type1 = 1
type2 = 2
type3 = 3
generate_csv(data_dir1, data_dir2, data_dir3, data_dir4, output_csv, type0, type1, type2, type3)