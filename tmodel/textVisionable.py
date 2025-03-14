import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

utils.logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_name = "/workplace/Llama3-Chinese-8B-Instruct-v3/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

medical_terms = [
    "adenocarcinoma",
    "dysphagia",
    "odynophagia",
    "arytenoid",
    "histopathological",
    "laryngoscopy",
    "sputum",
    "dyspnea",
    "gastroesophageal",
    "pathological"
]

num_added = tokenizer.add_tokens(
    [f"{term}" for term in medical_terms],
    special_tokens=False
)
print(f"Added {num_added} medical terms")

model.resize_token_embeddings(len(tokenizer))

input_text = " "

print("input_text:", input_text)
inputs = tokenizer(input_text, return_tensors='pt').to("cuda:0")
print("inputs:", inputs)

output_sequences = model.generate(
    **inputs,
    do_sample=True,
    max_length=inputs['input_ids'][0].shape[0]+1,
    temperature=0.9,
    top_p=0.6,
    repetition_penalty=1.1
)

output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("output_text:",output_text)

inputs = tokenizer(output_text, return_tensors='pt').to("cuda:0")
out = model(**inputs, output_attentions=True)

attention = out['attentions']
print("attention shape:", len(attention))
print("attention[0] shape:", attention[0].shape)

dropped_attention = tuple([layer[:,:,2:,2:] for layer in attention])
print("dropped_attention shape:", len(dropped_attention))
print("dropped_attention[0] shape:", dropped_attention[0].shape)

print("ids:", inputs['input_ids'])
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("tokens:", tokens)

dropped_tokens = tokens[2:]
print("dropped_tokens:", dropped_tokens)

selected_layer = 0

layer_attention = dropped_attention[selected_layer][0]

mean_attention = layer_attention.mean(axis=0)

global_attention = mean_attention.sum(axis=0)
print(global_attention)

global_attention_np = global_attention.detach().float().cpu().numpy()

input_ids = inputs['input_ids'][0].cpu().numpy()
tokens = [tokenizer.decode([input_id], skip_special_tokens=True) for input_id in input_ids]

plt.figure(figsize=(30, 2))
sns.heatmap(global_attention_np.reshape(1, -1),
            annot=False,
            cmap="YlOrRd",
            yticklabels=False,
            xticklabels=tokens,
            cbar=True)

plt.xticks(rotation=90)

plt.title("Global Attention Heatmap")
plt.tight_layout()
plt.savefig("results/test/t/attention.png")

