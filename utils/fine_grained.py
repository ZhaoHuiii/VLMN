import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/workplace/Llama3-Chinese-8B-Instruct-v3/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, do_sample=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data("Datasets/manVsModalData/csv/mvm.csv")

question = """
1. 患者的性别是？
2. 患者的年龄是？
3. 患者是否存在声嘶或咽异物感的症状？如有，该症状出现有多久了？呈持续性还是间歇性？是否反复？有无加重？声嘶后能否缓解？
4. 患者是否有反酸、嗳气的症状？
5. 患者是否有痰中带血的症状？
6. 患者是否有呼吸困难的症状？
7. 患者是否有吞咽困难的症状？
8. 患者是否曾做过喉镜检查？检查所见为？是否发现声带肿物？肿物表面是否粗糙？侵及范围如何？
9. 患者是否做过组织病理学检查？检查结果为？或者是否提示（轻/中/重度）不典型增生/原位癌/浸润癌？
10. 患者既往有无咽喉、食道、胃等消化系统方面的疾病？
11. 患者是否吸烟或饮酒，如有则频率如何？持续多久了？
12. 患者近期有无体重显著下降的情况？
13. 患者是否有其他不适或者需要注意的情况？
"""

prompt = """
<|begin_of_text|><|start_header_id|>system: <|end_header_id|>你是一个医术高超的医生。<|eot_id|>
<|start_header_id|>user: <|end_header_id|>以下是患者的病历信息，请根据这些信息回答相关问题：
{text}
{question}
回答的内容无需其他额外信息，只需回答问题即可，例如：
1. 男
2. 45岁
3. 有声嘶症状，持续2周，反复出现，无加重，声嘶后能缓解
4. 有反酸、嗳气症状
5. 无痰中带血症状
6. 有呼吸困难症状
7. 无吞咽困难症状
8. 未做过喉镜检查
9. 未做过组织病理学检查
10. 无消化系统疾病
11. 不吸烟，不饮酒
12. 无体重显著下降
13. 无其他不适
<|eot_id|>
<|start_header_id|>assistant: <|end_header_id|>
"""

def extract_fine_grained(generated_text):
    pattern = r'assistant: \s*(.*)'
    extract_text = re.search(pattern, generated_text, re.DOTALL).group()
    pattern = r'\d+\.\s*(.*?)\n'
    result = re.findall(pattern, extract_text)
    return result

def generate_text_from_model(text):

    formatted_prompt = prompt.format(text=text, question=question)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=1500).to(device)

    with torch.no_grad():

        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 512,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.15,
            top_k=2,
            top_p=0.98
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    fine_grained_texts = extract_fine_grained(generated_text)
    fine_grained_texts = "".join(fine_grained_texts)
    return fine_grained_texts

df['fine_grained_text'] = np.nan

df['fine_grained_text'] = df['fine_grained_text'].astype(str)

cache = {}

for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    text = row['text']
    
    if text in cache:
        generated = cache[text]
    else:
        generated = generate_text_from_model(text)
        cache[text] = generated
    
    df.at[idx, 'fine_grained_text'] = str(generated)

df.to_csv("Datasets/manVsModalData/csv/mvm_fine.csv", index=False)

print("over")