import pandas as pd
import os
import shutil

def process_internal_sheet(df, base_path):

    patients = []
    for _, row in df.iterrows():
        relative_path = row['姓名'].strip()
        full_path = os.path.join(base_path, relative_path)
        
        if not os.path.isdir(full_path):
            print(f"[内部表] 路径不存在: {full_path}")
            continue

        path_parts = relative_path.split('/')
        first_dir = path_parts[0]
        
        if first_dir in ['T1', 'T2', 'T3', 'T4']:
            category = 'Cancer'
        elif first_dir == 'WhitePatch':
            category = 'WhitePatch'
        else:
            print(f"[内部表] 未知分类目录: {first_dir}，路径: {relative_path}")
            continue

        patients.append((full_path, category))
    return patients

def process_city_sheet(df, base_path, city_name):
    patients = []
    for _, row in df.iterrows():
        name = row['姓名'].strip()
        found = False
        
        for case_type in ['WhitePatch', 'Cancer']:
            case_path = os.path.join(base_path, case_type)
            if not os.path.exists(case_path):
                continue

            for folder in os.listdir(case_path):
                folder_path = os.path.join(case_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                expected_txt = f"{folder}_姓名.txt"
                txt_path = os.path.join(folder_path, expected_txt)
                
                if os.path.isfile(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        file_content = f.read().strip()
                    
                    if file_content == name:
                        patients.append((folder_path, case_type))
                        found = True
                        print(f"[{city_name}] 找到患者: {name} 在 {case_type}/{folder}")
                        break
            if found:
                break
        
        if not found:
            print(f"[{city_name}] 未找到患者: {name}")
    return patients

def copy_and_rename_files(src_folder, dest_folder, old_num, new_num):
    for root, dirs, files in os.walk(src_folder):
        relative_path = os.path.relpath(root, src_folder)
        new_root = os.path.join(dest_folder, relative_path)
        os.makedirs(new_root, exist_ok=True)

        dir_basename = os.path.basename(root)
        if dir_basename.startswith(f"{old_num}_"):
            new_dir_name = dir_basename.replace(f"{old_num}_", f"{new_num}_", 1)
            new_root = os.path.join(os.path.dirname(new_root), new_dir_name)
            os.makedirs(new_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            
            if file.startswith(f"{old_num}_"):
                new_file = file.replace(f"{old_num}_", f"{new_num}_", 1)
            elif file.startswith(f"{old_num}."):
                new_file = file.replace(f"{old_num}.", f"{new_num}.", 1)
            else:
                new_file = file
            
            dest_file = os.path.join(new_root, new_file)
            
            shutil.copy2(src_file, dest_file)

def copy_patients(patients, output_dir):
    counters = {'Cancer': 0, 'WhitePatch': 0}
    
    for category in ['Cancer', 'WhitePatch']:
        category_path = os.path.join(output_dir, category)
        os.makedirs(category_path, exist_ok=True)
    
    for src_path, category in patients:
        counters[category] += 1
        new_num = str(counters[category])
        old_num = os.path.basename(src_path)
        
        dest_folder = os.path.join(output_dir, category, new_num)
        
        try:
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            
            os.makedirs(dest_folder)
            
            copy_and_rename_files(src_path, dest_folder, old_num, new_num)
            
            print(f"已处理 [{category}] 患者: {old_num} -> {new_num}")
        except Exception as e:
            print(f"处理失败 [{category}] {src_path} | 错误: {str(e)}")
    
    print("\n复制完成统计:")
    print(f"恶性患者(Cancer): {counters['Cancer']} 例")
    print(f"良性患者(WhitePatch): {counters['WhitePatch']} 例")

def main():
    excel_path = "Datasets/manVsModalData/manVsModal.xlsx"
    base_dir = "/data"
    
    internal_base = os.path.join(base_dir, "handleData")
    foshan_base = os.path.join(base_dir, "foShanData")
    zhaoqing_base = os.path.join(base_dir, "ZhaoQinData")
    output_dir = os.path.join(base_dir, "manVsModalData")

    try:
        df_internal = pd.read_excel(excel_path, sheet_name='内部')
        df_foshan = pd.read_excel(excel_path, sheet_name='佛山')
        df_zhaoqing = pd.read_excel(excel_path, sheet_name='肇庆')

        internal_patients = process_internal_sheet(df_internal, internal_base)
        foshan_patients = process_city_sheet(df_foshan, foshan_base, "佛山")
        zhaoqing_patients = process_city_sheet(df_zhaoqing, zhaoqing_base, "肇庆")

        all_patients = internal_patients + foshan_patients + zhaoqing_patients
        print(f"\n总共发现患者: {len(all_patients)} 例")

        copy_patients(all_patients, output_dir)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()