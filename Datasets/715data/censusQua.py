import pandas as pd
from collections import defaultdict

def count_unique_patients(csv_path):

    df = pd.read_csv(csv_path)
    
    patient_count = defaultdict(int)
    
    for path in df['image_path']:
        parts = path.split('/')
        if len(parts) >= 5:
            patient_id = f"{parts[3]}/{parts[4]}"
            patient_count[patient_id] += 1
    
    unique_patient_count = len(patient_count)
    
    return unique_patient_count, dict(patient_count)

if __name__ == "__main__":
    csv_path = "Datasets/715data/test_2.csv"
    unique_count, patient_details = count_unique_patients(csv_path)
    
    print(f"CSV textile patients number(only): {unique_count}")