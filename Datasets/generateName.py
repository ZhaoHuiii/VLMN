import os
import csv
from collections import OrderedDict

def extract_patient_name(patient_id, base_path="/data/handleData"):

    try:
        rel_path = patient_id.replace("\\", "/")
        dir_path = os.path.join(base_path, rel_path)
        
        dir_name = os.path.basename(dir_path)
        target_file = f"{dir_name}_患者姓名.txt"
        
        name_file = os.path.join(dir_path, target_file)
        
        with open(name_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
            
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return "Unknown"

def process_csv(input_path, output_path):

    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['name']
        
        ordered_fieldnames = list(OrderedDict.fromkeys(fieldnames))
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=ordered_fieldnames)
            writer.writeheader()
            
            for row in reader:
                patient_id = row['patient_id']
                
                patient_name = extract_patient_name(patient_id)
                
                new_row = row.copy()
                new_row['name'] = patient_name
                
                writer.writerow(new_row)

if __name__ == "__main__":
    process_csv('Datasets/715data/valid_1.csv', 'Datasets/715data/valid_1_withName.csv')
    print("处理完成，结果已保存到 output_with_name.csv")