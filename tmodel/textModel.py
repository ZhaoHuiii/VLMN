import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class ImageTextDataset(Dataset):
    def __init__(self, data, llama_tokenizer):
        self.data = data
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, label, image_path = row['fine_grained'], row['type'], row['image_path']
        text_inputs = self.llama_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}  # Remove batch dimension
        return text_inputs, torch.tensor(label, dtype=torch.long), image_path

def pad_collate(batch):
    max_text_input_ids_length = max(len(sample[0]['input_ids']) for sample in batch)
    max_text_attention_mask_length = max(len(sample[0]['attention_mask']) for sample in batch)

    text_input_ids = []
    text_attention_masks = []
    labels = []
    image_paths = []

    for sample in batch:
        text_input_ids.append(F.pad(sample[0]['input_ids'], (0, max_text_input_ids_length - len(sample[0]['input_ids']))))
        text_attention_masks.append(F.pad(sample[0]['attention_mask'], (0, max_text_attention_mask_length - len(sample[0]['attention_mask']))))
        labels.append(sample[1])
        image_paths.append(sample[2])

    text_input_ids = torch.stack(text_input_ids)
    text_attention_masks = torch.stack(text_attention_masks)
    labels = torch.stack(labels)

    return {'input_ids': text_input_ids, 'attention_mask': text_attention_masks}, labels, image_paths

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop_duplicates(subset=['fine_grained'], keep='first')
    return data

train_data = load_data('Datasets/715data/train_7.csv')
val_data = load_data('Datasets/715data/valid_1.csv')
test_data = load_data('Datasets/715data/test_2.csv')

def print_class_distribution(df, name):
    counts = df['type'].value_counts().sort_index()
    print(f"\n{name}类别分布:")
    print(f"类别 0: {counts.get(0, 0)} 个样本")
    print(f"类别 1: {counts.get(1, 0)} 个样本")
    print(f"总计: {len(df)} 个样本")
    print(f"比例: 0类占 {counts.get(0, 0)/len(df):.1%}, 1类占 {counts.get(1, 0)/len(df):.1%}")

print_class_distribution(test_data, "测试集")

print(f"训练集样本数: {len(train_data)}, 验证集样本数: {len(val_data)}, 测试集样本数: {len(test_data)}")

llama_tokenizer = AutoTokenizer.from_pretrained("/home/jzh/workplace/Llama3-Chinese-8B-Instruct-v3/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3")

llama_tokenizer.pad_token = llama_tokenizer.eos_token

train_dataset = ImageTextDataset(train_data, llama_tokenizer)
val_dataset = ImageTextDataset(val_data, llama_tokenizer)
test_dataset = ImageTextDataset(test_data, llama_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

device_llama = torch.device("cuda:0")
device_cal = torch.device("cuda:0")

llama_model = AutoModelForCausalLM.from_pretrained("/home/jzh/workplace/Llama3-Chinese-8B-Instruct-v3/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3", torch_dtype=torch.float32).to(device_llama)

text_hidden_size = llama_model.config.hidden_size

class MultiModalClassifier(torch.nn.Module):
    def __init__(self, llama_model, text_hidden_size, num_classes, dropout_prob=0.5):
        super(MultiModalClassifier, self).__init__()
        self.llama_model = llama_model
        
        for param in self.llama_model.parameters():
            param.requires_grad = False
        
        self.text_fc = torch.nn.Linear(text_hidden_size, 2048).to(device_llama)
        
        self.fc1 = torch.nn.Linear(2048, 1024).to(device_cal)
        self.dropout = torch.nn.Dropout(dropout_prob).to(device_cal)
        self.fc2 = torch.nn.Linear(1024, num_classes).to(device_cal)
    
    def forward(self, text_inputs):
        text_inputs = {k: v.to(device_llama) for k, v in text_inputs.items()}
        text_outputs = self.llama_model(**text_inputs, output_hidden_states=True)
        text_hidden_states = text_outputs.hidden_states[-1]
        pooled_text_output = torch.mean(text_hidden_states, dim=1).float().to(device_llama)
        
        reduced_text_output = F.relu(self.text_fc(pooled_text_output)).to(device_cal)
        
        x = F.relu(self.fc1(reduced_text_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

num_classes = 2
classifier = MultiModalClassifier(llama_model, text_hidden_size, num_classes)

criterion = torch.nn.CrossEntropyLoss().to(device_cal)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

def plot_confusion_matrix(all_test_labels, all_test_preds_labels, resultPath, timestamp):
    cm = confusion_matrix(all_test_labels, all_test_preds_labels)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Dysplasia", "Carcinoma"])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='.2f', colorbar=True)

    plt.yticks(rotation=90)
    
    plt.xlabel('Prediction')
    plt.ylabel('Diagnosis')
    
    plt.title('Text-based Monomodal model')
    
    plt.savefig(f'{resultPath}/confusion_matrix.png')
    plt.close()

def save_roc_curve_values(fpr, tpr, thresholds, resultPath, timestamp):
    roc_values_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds
    })
    roc_values_df.to_csv(f'{resultPath}/roc_curve_values.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def save_dca_values(model_name, thresh_group, net_benefit_model, net_benefit_all, resultPath, timestamp):
    dca_values_df = pd.DataFrame({
        'Threshold': thresh_group,
        'Net Benefit Model': net_benefit_model,
        'Net Benefit All': net_benefit_all
    })
    dca_values_df.to_csv(f'{resultPath}/dca_values_{model_name}_{timestamp}.csv', index=False)

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):

    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh + 1e-10))  # 避免除以零
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):

    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh + 1e-10))  # 避免除以零
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(y_true, y_pred, resultPath, timestamp, model_name):

    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred, y_true)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_true)

    save_dca_values(model_name, thresh_group, net_benefit_model, net_benefit_all, resultPath, timestamp)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresh_group, net_benefit_model, color='crimson', label=f'Model: {model_name}')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat All')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat None')

    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)
    ax.set_xlabel('Threshold Probability', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    ax.set_ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    ax.grid(True)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')

    plt.savefig(f'{resultPath}/dca_curve_{model_name}_{timestamp}.png', dpi=300)
    plt.close()

def train_model(model, train_loader, val_loader, test_loader, resultPath, criterion, optimizer, scheduler, num_epochs=5):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resultPath = resultPath + "/" + timestamp
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    smoothed_train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for text_inputs, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            text_inputs = {k: v.to(device_llama) for k, v in text_inputs.items()}
            labels = labels.to(device_cal)

            optimizer.zero_grad()
            outputs = model(text_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        smooth_loss = sum(train_losses[-10:]) / min(len(train_losses), 10)
        smoothed_train_losses.append(smooth_loss)

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Learning Rate: {scheduler.get_last_lr()}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for text_inputs, labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                text_inputs = {k: v.to(device_llama) for k, v in text_inputs.items()}
                labels = labels.to(device_cal)

                outputs = model(text_inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_accuracies.append(val_acc.item())
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{resultPath}/best_model.pth')

        torch.cuda.empty_cache()

    test_corrects = 0
    all_test_labels = []
    all_test_preds = []
    all_patient_paths = []

    with torch.no_grad():
        for text_inputs, labels, patient_paths in tqdm(test_loader, desc="Testing"):
            text_inputs = {k: v.to(device_llama) for k, v in text_inputs.items()}
            labels = labels.to(device_cal)

            outputs = model(text_inputs)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            _, preds = torch.max(outputs, 1)

            test_corrects += torch.sum(preds == labels.data)

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(probabilities.cpu().numpy())
            all_patient_paths.extend(patient_paths)

    df = pd.DataFrame({
        'patient_path': all_patient_paths,
        'true_labels': all_test_labels,
        'predicted_labels': all_test_preds
    })
    df.to_csv(f'{resultPath}/test_results.csv', index=False)

    threshold = 0.5
    all_test_preds_labels = [1 if prob >= threshold else 0 for prob in all_test_preds]  

    test_acc = test_corrects.double() / len(test_loader.dataset)
    f1 = f1_score(all_test_labels, all_test_preds_labels, average='weighted')
    recall = recall_score(all_test_labels, all_test_preds_labels, average='weighted')
    precision = precision_score(all_test_labels, all_test_preds_labels, average='weighted')

    fpr, tpr, thresholds = roc_curve(all_test_labels, all_test_preds)
    roc_auc = auc(fpr, tpr)

    save_roc_curve_values(fpr, tpr, thresholds, resultPath, timestamp)

    precision_vals, recall_vals, _ = precision_recall_curve(all_test_labels, all_test_preds)

    print(f"Test Accuracy: {test_acc}")
    print(f"F1 Score: {f1}, Recall: {recall}, Precision: {precision}")

    with open(os.path.join(resultPath, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs+1), smoothed_train_losses, label='Train Loss (Smoothed)')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs+1), learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{resultPath}/training_results_{timestamp}.png')
    plt.close()

    plot_confusion_matrix(all_test_labels, all_test_preds_labels, resultPath, timestamp)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(f'{resultPath}/roc_curve_{timestamp}.png')
    plt.close()

    plt.figure()
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f'{resultPath}/pr_curve_{timestamp}.png')
    plt.close()

    plot_DCA(np.array(all_test_labels), np.array(all_test_preds), resultPath, timestamp, "Text-based Monomodal model")

    return model

resPath = "results/text_results"
train_model(classifier, train_loader, val_loader, test_loader, resPath, criterion, optimizer, scheduler, num_epochs=40)