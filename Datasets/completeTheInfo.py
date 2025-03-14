import csv
import os

def update_csv(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = row['image_path']
            path_seg = image_path.split('/')[:-2]
            patient_id = path_seg[4]

            folder_path = f"/{path_seg[1]}/{path_seg[2]}/{path_seg[3]}/{patient_id}"
            
            history_file = os.path.join(folder_path, f"{patient_id}_既往史.txt")
            personal_file = os.path.join(folder_path, f"{patient_id}_个人史.txt")
            
            history_text = ""
            personal_text = ""

            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as hf:
                    history_text = hf.read().strip()
            
            if os.path.exists(personal_file):
                with open(personal_file, 'r', encoding='utf-8') as pf:
                    personal_text = pf.read().strip()
            
            row['text'] += f"既往史: {history_text} 个人史: {personal_text}"
            rows.append(row)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        fieldnames = ['image_path', 'text', 'type', 'patient_id']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    csv_path = '/home/jzh/workplace_CT/Datasets/outData/zq/zq copy.csv'
    update_csv(csv_path)