import os
import pandas as pd

def process_patient_folder(patient_folder, type):
    data = []
    laryngoscope_folder = os.path.join(patient_folder, '喉镜')

    if not os.path.isdir(laryngoscope_folder):
        return data

    baseInfo_file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_基本信息.txt")
    principleAction_file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_主诉.txt")
    nowHistory_file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_现病史.txt")
    
    if os.path.exists(baseInfo_file_path) or os.path.exists(principleAction_file_path) or os.path.exists(nowHistory_file_path):
        with open(baseInfo_file_path, 'r', encoding='utf-8') as f:
            baseInfo = f.read().strip()
        with open(principleAction_file_path, 'r', encoding='utf-8') as f:
            principleAction = f.read().strip()
        with open(nowHistory_file_path, 'r', encoding='utf-8') as f:
            nowHistory = f.read().strip()
    else:
        return data

    description = baseInfo + " 主诉：" + principleAction + " 现病史：" + nowHistory

    for root, dirs, files in os.walk(laryngoscope_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                data.append({'image_path': image_path, 'text': description, 'type': type})

    return data

def generate_csv(data_dir1, data_dir2, data_dir5, output_csv, type0, type1):
    all_data = []

    for patient_folder in os.listdir(data_dir1):
        patient_path = os.path.join(data_dir1, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type1)
            all_data.extend(patient_data)
    
    for patient_folder in os.listdir(data_dir2):
        patient_path = os.path.join(data_dir2, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type1)
            all_data.extend(patient_data)

    for patient_folder in os.listdir(data_dir5):
        patient_path = os.path.join(data_dir5, patient_folder)
        if os.path.isdir(patient_path):
            patient_data = process_patient_folder(patient_path, type0)
            all_data.extend(patient_data)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)

data_dir1 = 'Datasets/processData_T1'
data_dir2 = 'Datasets/processData_T2'
data_dir5 = 'Datasets/processData_WhitePatch'
output_csv = 'Datasets/it_t1t2.csv'
type0 = 0
type1 = 1

generate_csv(data_dir1, data_dir2, data_dir5, output_csv, type0, type1)
