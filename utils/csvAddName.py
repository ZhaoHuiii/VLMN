import pandas as pd
import os

def extract_patient_name(patient_path):
    try:
        base_dir = os.path.dirname(os.path.dirname(patient_path))

        patient_id = os.path.basename(base_dir)
        
        possible_files = [
            os.path.join(base_dir, f"{patient_id}_患者姓名.txt"),
            os.path.join(base_dir, f"{patient_id}_姓名.txt")
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        
        return "Name_Not_Found"
    except:
        return "Error_Extracting_Name"

df = pd.read_csv("results/rj_results/it/20250225_133847/test_results.csv")

df["patient_name"] = df["patient_path"].apply(extract_patient_name)

df.to_csv("results/rj_results/it/20250225_133847/test_results_withName.csv", index=False)
print("over.")