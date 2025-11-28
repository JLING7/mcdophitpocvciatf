import os
import time
import random
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from torchvision import models, transforms
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from single_model import convert_ln_to_dyt, MDST_EfficientNet


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def analyze_model_performance(model, input_size=(1, 3, 512, 512), device='cuda', warmup=10, runs=100):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    input_tensor = torch.randn(*input_size).to(device)

  
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        print("FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
        print("Params:")
        print(parameter_count_table(model))

 
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_tensor)
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
    print("Peak memory usage: {:.2f} MB".format(peak_memory))


    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
    latency = (end - start) / runs * 1000  # 毫秒
    print("Average latency: {:.2f} ms".format(latency))


class CTImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
    
        for fname in os.listdir(img_dir):
            if fname.endswith(".png"):
           
                if fname.startswith("4"):
                    label = 0
                elif fname.startswith(("1", "2", "3")):
                    label = 1
                else:
                    continue  
                self.image_paths.append(os.path.join(img_dir, fname))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor()    
])
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()    
])


train_dir = r"four_class/data/other/train"  
val_dir = r"four_class/data/other/val"  
test_dir = r"four_class/data/other/test"    

train_dataset = CTImageDataset(train_dir, transform=transform)
val_dataset = CTImageDataset(val_dir, transform=transform1)
test_dataset = CTImageDataset(test_dir, transform=transform1)

# 可以接着使用 DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 训练模型
num_epochs = 40
lr = 1e-4
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, roc_auc_score


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 1
model = MDST_EfficientNet(num_classes =num_classes)
model = convert_ln_to_dyt(model)


model = model.to(device)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total/1e6:.3f}M, 可训练参数量: {trainable/1e6:.3f}M")
#analyze_model_performance(model)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=lr * 0.1)
best_auc = 0
best_f1 = 0
best_acc = 0
patience = 10
counter = 0

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    all_train_labels = []
    all_train_preds = []
    
    for imgs, e_data in train_loader:
        imgs, e_data = imgs.to(device), e_data.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, e_data.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

 
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_train_labels.extend(e_data.cpu().numpy())
        all_train_preds.extend(preds)


    all_train_labels = np.array(all_train_labels)
    all_train_preds = np.array(all_train_preds)
    train_accuracy = accuracy_score(all_train_labels, (all_train_preds > 0.5).astype(int))
    train_recall = recall_score(all_train_labels, (all_train_preds > 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(all_train_labels, (all_train_preds > 0.5).astype(int)).ravel()
    train_specificity = tn / (tn + fp)
    train_ppv = precision_score(all_train_labels, (all_train_preds > 0.5).astype(int))
    train_npv = tn / (tn + fn)
    train_f1 = f1_score(all_train_labels, (all_train_preds > 0.5).astype(int))
    train_auc = roc_auc_score(all_train_labels, all_train_preds)


    model.eval()
    val_loss = 0.0
    all_val_labels = []
    all_val_preds = []
    
    with torch.no_grad():
        for imgs, e_data in val_loader:
            imgs, e_data = imgs.to(device), e_data.to(device)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs)
            all_val_labels.extend(e_data.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
            loss = criterion(outputs, e_data.unsqueeze(1))
            val_loss += loss.item()


    all_val_labels = np.array(all_val_labels)
    all_val_preds = np.array(all_val_preds)
    val_accuracy = accuracy_score(all_val_labels, (all_val_preds > 0.5).astype(int))
    val_recall = recall_score(all_val_labels, (all_val_preds > 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(all_val_labels, (all_val_preds > 0.5).astype(int)).ravel()
    val_specificity = tn / (tn + fp)
    val_ppv = precision_score(all_val_labels, (all_val_preds > 0.5).astype(int))
    val_npv = tn / (tn + fn)
    val_f1 = f1_score(all_val_labels, (all_val_preds > 0.5).astype(int))
    val_auc = roc_auc_score(all_val_labels, all_val_preds)



    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, Specificity: {train_specificity:.4f}, PPV: {train_ppv:.4f}, NPV: {train_npv:.4f}, F1 Score: {train_f1:.4f}, AUC: {train_auc:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}, PPV: {val_ppv:.4f}, NPV: {val_npv:.4f}, F1 Score: {val_f1:.4f}, AUC: {val_auc:.4f}")
    
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "weight.pth") 
        counter = 0
    else:
        counter += 1
        print(f"No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered!")
            break

model.load_state_dict(torch.load("weight.pth"))
model.eval()
test_loss = 0.0
all_test_labels = []
all_test_preds = []
    
with torch.no_grad():
    for imgs, e_data in test_loader:
        imgs, e_data = imgs.to(device), e_data.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs)
        all_test_labels.extend(e_data.cpu().numpy())
        all_test_preds.extend(preds.cpu().numpy())
        loss = criterion(outputs, e_data.unsqueeze(1))
        val_loss += loss.item()


all_test_labels = np.array(all_test_labels)
all_test_preds = np.array(all_test_preds)
test_accuracy = accuracy_score(all_test_labels, (all_test_preds > 0.5).astype(int))
test_recall = recall_score(all_test_labels, (all_test_preds > 0.5).astype(int))
tn, fp, fn, tp = confusion_matrix(all_test_labels, (all_test_preds > 0.5).astype(int)).ravel()
test_specificity = tn / (tn + fp)
test_ppv = precision_score(all_test_labels, (all_test_preds > 0.5).astype(int))
test_npv = tn / (tn + fn)
test_f1 = f1_score(all_test_labels, (all_test_preds > 0.5).astype(int))
test_auc = roc_auc_score(all_test_labels, all_test_preds)

print(f"Test Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, Specificity: {test_specificity:.4f}, PPV: {test_ppv:.4f}, NPV: {test_npv:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")



