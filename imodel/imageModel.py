import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class ImageTextDataset(Dataset):
    def __init__(self, data, blip_processor):
        self.data = data
        self.blip_processor = blip_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path, label = row['image_path'], row['type']
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.blip_processor(images=image, return_tensors="pt")
        image_inputs = {k: v.squeeze(0) for k, v in image_inputs.items()}  # Remove batch dimension
        return image_inputs, torch.tensor(label, dtype=torch.long), image_path

def pad_collate(batch):
    max_pixel_values_length = max(sample[0]['pixel_values'].shape[1] for sample in batch)

    pixel_values = []
    labels = []
    image_paths = []

    for sample in batch:
        pixel_values.append(torch.nn.functional.pad(sample[0]['pixel_values'], (0, 0, 0, max_pixel_values_length - sample[0]['pixel_values'].shape[1])))
        labels.append(sample[1])
        image_paths.append(sample[2])

    pixel_values = torch.stack(pixel_values)
    labels = torch.stack(labels)

    return {'pixel_values': pixel_values}, labels, image_paths

def load_data(file_path):
    return pd.read_csv(file_path)

train_data = load_data('Datasets/715data/train_7.csv') 
val_data = load_data('Datasets/715data/valid_1.csv') 
test_data = load_data('Datasets/715data/test_2.csv') 

def filter_t3_t4(data):
    return data[~data['image_path'].str.contains('data/handleData/T3|data/handleData/T4')]

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

train_dataset = ImageTextDataset(train_data, blip_processor)
val_dataset = ImageTextDataset(val_data, blip_processor)
test_dataset = ImageTextDataset(test_data, blip_processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

device_blip = torch.device("cuda:1")
device_cal = torch.device("cuda:1")

blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32).to(device_blip)

vision_hidden_size = blip_model.config.vision_config.hidden_size

class MultiModalClassifier(torch.nn.Module):
    def __init__(self, blip_model, vision_hidden_size, num_classes, dropout_prob=0.5):
        super(MultiModalClassifier, self).__init__()

        self.img_model = blip_model.vision_model

        for param in self.img_model.parameters():
            param.requires_grad = False
        
        self.vision_fc = torch.nn.Linear(vision_hidden_size, 2048).to(device_blip)
        
        self.fc1 = torch.nn.Linear(2048, 1024).to(device_cal)
        self.dropout = torch.nn.Dropout(dropout_prob).to(device_cal)
        self.fc2 = torch.nn.Linear(1024, num_classes).to(device_cal)
    

    def forward(self, image_inputs):

        reduced_vision_output = self.img_encoder(img_inputs=image_inputs)

        x = F.relu(self.fc1(reduced_vision_output))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
    

    def img_encoder(self, img_inputs):

        image_inputs = {k: v.to(device_blip) for k, v in img_inputs.items()}
        vision_outputs = self.img_model(**image_inputs)
        vision_hidden_states = vision_outputs.last_hidden_state
        pooled_vision_output = torch.mean(vision_hidden_states, dim=1).float()

        reduced_vision_output = F.relu(self.vision_fc(pooled_vision_output)).to(device_cal)

        return reduced_vision_output

num_classes = 2
classifier = MultiModalClassifier(blip_model, vision_hidden_size, num_classes)

criterion = torch.nn.CrossEntropyLoss().to(device_cal)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)

def plot_confusion_matrix(all_test_labels, all_test_preds_labels, resultPath, timestamp):
    cm = confusion_matrix(all_test_labels, all_test_preds_labels)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Dysplasia", "Carcinoma"])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='.2f', colorbar=True)

    plt.yticks(rotation=90)
    
    plt.xlabel('Prediction')
    plt.ylabel('Diagnosis')
    
    plt.title('Image-based Monomodal model')
    
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
        loss = 0.0
        loss_total = 0.0
        
        for image_inputs, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            image_inputs = {k: v.to(device_blip) for k, v in image_inputs.items()}
            labels = labels.to(device_cal)

            optimizer.zero_grad()
            outputs = model(image_inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss += loss.item()
            loss_total += loss.item()

        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])

        train_loss = loss / len(train_loader)
        train_losses.append(train_loss)

        smooth_loss = sum(train_losses[-10:]) / min(len(train_losses), 10)
        smoothed_train_losses.append(smooth_loss)

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Contrastive Loss: {loss_total / len(train_loader)}, Learning Rate: {scheduler.get_last_lr()}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for image_inputs, labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                image_inputs = {k: v.to(device_blip) for k, v in image_inputs.items()}
                labels = labels.to(device_cal)

                logits = model(image_inputs)

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, preds = torch.max(logits, 1)
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
        for image_inputs, labels, patient_paths in tqdm(test_loader, desc="Testing"):
            image_inputs = {k: v.to(device_blip) for k, v in image_inputs.items()}
            labels = labels.to(device_cal)

            logits = model(image_inputs)

            probabilities = torch.softmax(logits, dim=1)[:, 1]

            _, preds = torch.max(logits, 1)

            test_corrects += torch.sum(preds == labels.data)

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(probabilities.cpu().numpy())
            all_patient_paths.extend(patient_paths)

    df = pd.DataFrame({
        'patient_path': all_patient_paths,
        'true_labels': all_test_labels,
        'predicted_labels': all_test_preds
    })
    df.to_csv(f'{resPath}/test_results.csv', index=False)

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
    plt.plot(range(1, num_epochs+1), [loss.detach().cpu().numpy() for loss in smoothed_train_losses], label='Train Loss (Smoothed)')
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

    plot_DCA(np.array(all_test_labels), np.array(all_test_preds), resultPath, timestamp, "Image-based Monomodal model")

    return model

resPath = "results"  
train_model(classifier, train_loader, val_loader, test_loader, resPath, criterion, optimizer, scheduler, num_epochs=20)