import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('Datasets/715data/it715.csv')

def extract_patient_id(image_path):
    try:
        patient_id = os.path.dirname(image_path).split('/')[3] + '/' + os.path.dirname(image_path).split('/')[4]
        print(patient_id)
        return patient_id
    except Exception as e:
        print(f"Error parsing image path {image_path}: {e}")
        return None

df['patient_id'] = df['image_path'].apply(extract_patient_id)

patient_ids = df['patient_id'].unique()

train_patient_ids, temp_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=1)
valid_patient_ids, test_patient_ids = train_test_split(temp_patient_ids, test_size=0.5, random_state=1)

train_df = df[df['patient_id'].isin(train_patient_ids)]
valid_df = df[df['patient_id'].isin(valid_patient_ids)]
test_df = df[df['patient_id'].isin(test_patient_ids)]

train_df.to_csv('Datasets/715data/train_ori.csv', index=False)
valid_df.to_csv('Datasets/715data/valid_ori.csv', index=False)
test_df.to_csv('Datasets/715data/test_ori.csv', index=False)

print("over.")
