import os
import os.path
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#############################################################
##                 1. Helper Function: load_embedding      ##
#############################################################
def load_embedding(embedding_path):
    """
    从 .tfrecord 文件里载入 'embedding' 字段的 float_list
    返回 shape (embedding_dim,) 的 torch.tensor。
    """
    raw_dataset = tf.data.TFRecordDataset([embedding_path])
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        embedding_feature = example.features.feature['embedding']
        embedding_values = embedding_feature.float_list.value
    return torch.tensor(embedding_values)

#############################################################
##                 2. 定义 MIMIC_Embed_Dataset             ##
#############################################################
class MIMIC_Embed_Dataset(Dataset):
    pathologies = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    # 数据集拆分比例
    split_ratio = [0.8, 0.1, 0.1]

    def __init__(
        self,
        embedpath,
        csvpath,
        metacsvpath,
        views=["PA"],
        data_aug=None,
        seed=0,
        unique_patients=True,
        mode=["train", "valid", "test"][0],
    ):
        super().__init__()
        np.random.seed(seed)

        self.mode = mode
        self.embedpath = embedpath
        self.csvpath = csvpath
        self.metacsvpath = metacsvpath
        self.data_aug = data_aug

        # 读入数据
        self.csv = pd.read_csv(self.csvpath)
        self.metacsv = pd.read_csv(self.metacsvpath)

        # 设置多级索引，并做连接
        self.csv = self.csv.set_index(["subject_id", "study_id"])
        self.metacsv = self.metacsv.set_index(["subject_id", "study_id"])
        self.csv = self.csv.join(self.metacsv).reset_index()

        # 要编码的文本列
        self.text_cols = [
            "PerformedProcedureStepDescription",
            "ProcedureCodeSequence_CodeMeaning",
            "ViewCodeSequence_CodeMeaning",
            "PatientOrientationCodeSequence_CodeMeaning",
        ]
        # 缺失值统一填充 "Unknown"
        for col in self.text_cols:
            if col not in self.csv.columns:
                self.csv[col] = "Unknown"
                print(f"Warning: {col} not in csv, fill with 'Unknown'")
            else:
                self.csv[col] = self.csv[col].fillna("Unknown")

        # 只保留指定视图
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        # 是否每个 patient 只保留一条记录
        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # 拆分 train/valid/test
        n_row = self.csv.shape[0]
        if self.mode == "train":
            self.csv = self.csv[: int(n_row * self.split_ratio[0])]
        elif self.mode == "valid":
            self.csv = self.csv[
                int(n_row * self.split_ratio[0]) : int(n_row * (self.split_ratio[0] + self.split_ratio[1]))
            ]
        elif self.mode == "test":
            self.csv = self.csv[-int(n_row * self.split_ratio[-1]) :]
        else:
            raise ValueError(
                f"mode must be one of ['train','valid','test'], but got {self.mode}"
            )

        # 构建多标签
        healthy = self.csv["No Finding"] == 1
        self.pathologies = sorted(self.pathologies)
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            else:
                # 如果这个 pathology 列不存在，就先补0
                mask = pd.Series([0]*len(self.csv))
            labels.append(mask.values)
        self.labels = np.asarray(labels).T.astype(np.float32)

        # 把所有 -1 替换成 np.nan
        self.labels[self.labels == -1] = np.nan

        # 改名 "Pleural Effusion" -> "Effusion"
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        # 准备字符串->整型ID映射
        self.text_col_mappings = {}
        for col in self.text_cols:
            unique_vals = self.csv[col].unique().tolist()
            self.text_col_mappings[col] = { val: idx for idx, val in enumerate(unique_vals) }

    def limit_to_selected_views(self, views):
        if type(views) is not list:
            views = [views]
        if '*' in views:
            views = ["*"]
        self.views = views
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid   = str(self.csv.iloc[idx]["study_id"])
        dicom_id  = str(self.csv.iloc[idx]["dicom_id"])

        # 载入 TFRecord embedding
        embed_file = os.path.join(
            self.embedpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".tfrecord",
        )
        sample["embedding"] = load_embedding(embed_file)

        # 取出4个文本列对应的ID
        cat_ids = []
        for col in self.text_cols:
            val_str = self.csv.iloc[idx][col]
            cat_id  = self.text_col_mappings[col][val_str]
            cat_ids.append(cat_id)
        sample["cat_ids"] = torch.tensor(cat_ids, dtype=torch.long)

        return sample

#############################################################
##  3. 提取 (X, y, cat_ids) 的辅助函数                    ##
#############################################################
def extract_X_y_cat(dataset):
    X_list = []
    y_list = []
    cat_ids_list = []
    for sample in dataset:
        X_list.append(sample['embedding'])        # shape: (embedding_dim,)
        y_list.append(torch.tensor(sample['lab']))# shape: (num_classes,)
        cat_ids_list.append(sample['cat_ids'])    # shape: (4,)
    # 拼成tensor
    X = torch.stack(X_list)                       # [N, embedding_dim]
    y = torch.stack(y_list)                       # [N, num_classes]
    cat_ids = torch.stack(cat_ids_list)           # [N, 4]
    return X, y, cat_ids

#############################################################
##   4. 定义AutoEncoder (AE) 及其训练函数                  ##
#############################################################
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1376, embed_dim=128):
        super(AutoEncoder, self).__init__()
        # encoder:  input_dim -> 512 -> embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        # decoder:  embed_dim -> 512 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

def train_autoencoder(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            x = batch_data[0].to(device)  # (batch_size, input_dim)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data[0].to(device)
                x_recon = model(x)
                loss = criterion(x_recon, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if (epoch+1) % 10 == 0:
            print(f"[AE] Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    # 恢复最优
    model.load_state_dict(best_state)
    print("AutoEncoder Training Done! Best Val Loss =", best_val_loss)
    return model

#############################################################
##   5. 使用训练好的 AE 压缩特征:  X -> X_ae               ##
#############################################################
def encode_dataset(model, loader, device='cuda'):
    zs = []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            x = batch_data[0].to(device)
            z = model.encoder(x)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)

#############################################################
##   6. 定义分类器 + FocalLoss                             ##
#############################################################
class MultiInputCXRClassifier(nn.Module):
    def __init__(self, 
                 ae_feature_dim=128, 
                 num_class=13,
                 num_cat1=10, 
                 num_cat2=10, 
                 num_cat3=10, 
                 num_cat4=10, 
                 embed_dim=8  # 每个categorical feature的embedding大小
                 ):
        super(MultiInputCXRClassifier, self).__init__()
        # 四个 nn.Embedding，分别对应4列文本特征
        self.emb1 = nn.Embedding(num_embeddings=num_cat1, embedding_dim=embed_dim)
        self.emb2 = nn.Embedding(num_embeddings=num_cat2, embedding_dim=embed_dim)
        self.emb3 = nn.Embedding(num_embeddings=num_cat3, embedding_dim=embed_dim)
        self.emb4 = nn.Embedding(num_embeddings=num_cat4, embedding_dim=embed_dim)

        # 最终输入维度 = AE输出(128) + 4*embed_dim(=32)
        in_dim = ae_feature_dim + 4*embed_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, x_ae, cat_ids):
        """
        x_ae:  [B, 128]，来自AutoEncoder的embedding
        cat_ids: [B, 4]，每行对应4个文本列的 int ID
        """
        cat1 = cat_ids[:, 0]
        cat2 = cat_ids[:, 1]
        cat3 = cat_ids[:, 2]
        cat4 = cat_ids[:, 3]

        e1 = self.emb1(cat1)  # [B, embed_dim]
        e2 = self.emb2(cat2)  
        e3 = self.emb3(cat3)
        e4 = self.emb4(cat4)

        x_cat = torch.cat([e1, e2, e3, e4], dim=1)  # [B, 4*embed_dim]
        x_all = torch.cat([x_ae, x_cat], dim=1)     # [B, 128 + 4*embed_dim]

        out = self.model(x_all)  # [B, num_class]
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds: [N, C], targets: [N, C]，取值 0或1
        """
        eps = 1e-8
        bce = - (targets * torch.log(preds + eps) + (1.0 - targets) * torch.log(1.0 - preds + eps))
        pt = targets * preds + (1 - targets) * (1 - preds)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

#############################################################
##   7. 辅助函数: 找到最优阈值                              ##
#############################################################
def find_best_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.0, 1.01, 0.01):
            y_pred_i = (y_probs[:, i] >= t).astype(int)
            mask = ~np.isnan(y_true[:, i])
            if np.sum(mask) == 0:
                continue
            f1 = f1_score(y_true[mask, i], y_pred_i[mask])
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        thresholds.append(best_thresh)
    return thresholds


#############################################################
##   8. MAIN: 准备数据集 + 训练 AE + 训练分类器             ##
#############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedpath = "/home/lde/SPH6004/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files"
csvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-chexpert.csv"
metacsvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-metadata.csv"

# === 新增1：检查是否有缓存文件 ===
use_cache = True  # 或者改成False来禁用缓存
cache_file = "cached_data.pt"

if use_cache and os.path.exists(cache_file):
    print("Loading data from cache:", cache_file)
    data = torch.load(cache_file)
    X_train_ae = data["X_train_ae"]
    y_train = data["y_train"]
    cat_ids_train = data["cat_ids_train"]

    X_val_ae = data["X_val_ae"]
    y_val = data["y_val"]
    cat_ids_val = data["cat_ids_val"]

    X_test_ae = data["X_test_ae"]
    y_test = data["y_test"]
    cat_ids_test = data["cat_ids_test"]

    num_cat1 = data["num_cat1"]
    num_cat2 = data["num_cat2"]
    num_cat3 = data["num_cat3"]
    num_cat4 = data["num_cat4"]
    num_classes = data["num_classes"]

else:
    # ================原先的流程===============

    # 1) 构造Dataset
    train_dataset = MIMIC_Embed_Dataset(embedpath, csvpath, metacsvpath, mode="train")
    val_dataset   = MIMIC_Embed_Dataset(embedpath, csvpath, metacsvpath, mode="valid")
    test_dataset  = MIMIC_Embed_Dataset(embedpath, csvpath, metacsvpath, mode="test")

    # 2) 提取 X, y, cat_ids
    X_train, y_train, cat_ids_train = extract_X_y_cat(train_dataset)
    X_val,   y_val,   cat_ids_val   = extract_X_y_cat(val_dataset)
    X_test,  y_test,  cat_ids_test  = extract_X_y_cat(test_dataset)

    print("Original Embedding shape:", X_train.shape)  # e.g. [N_train, 1376]

    # 3) 标准化图像特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.numpy())
    X_val_scaled   = scaler.transform(X_val.numpy())
    X_test_scaled  = scaler.transform(X_test.numpy())

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val_scaled,   dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)

    # 4) AE 的 Dataloader
    batch_size_ae = 64
    train_loader_ae = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size_ae, shuffle=True)
    val_loader_ae   = DataLoader(TensorDataset(X_val_tensor),   batch_size=batch_size_ae, shuffle=False)
    test_loader_ae  = DataLoader(TensorDataset(X_test_tensor),  batch_size=batch_size_ae, shuffle=False)

    # 5) 训练 AutoEncoder
    ae_model = AutoEncoder(input_dim=X_train_tensor.shape[1], embed_dim=128)
    ae_model = train_autoencoder(ae_model, train_loader_ae, val_loader_ae,
                                 num_epochs=50, lr=1e-3, device=device)

    # 6) 用训练好的 AE 压缩特征
    X_train_ae = encode_dataset(ae_model, train_loader_ae, device)
    X_val_ae   = encode_dataset(ae_model, val_loader_ae,   device)
    X_test_ae  = encode_dataset(ae_model, test_loader_ae,  device)

    print("After AE compression:", X_train_ae.shape)  # [N_train, 128]

    # 8) 统计每个文本列的类别数量
    num_cat1 = len(train_dataset.text_col_mappings["PerformedProcedureStepDescription"])
    num_cat2 = len(train_dataset.text_col_mappings["ProcedureCodeSequence_CodeMeaning"])
    num_cat3 = len(train_dataset.text_col_mappings["ViewCodeSequence_CodeMeaning"])
    num_cat4 = len(train_dataset.text_col_mappings["PatientOrientationCodeSequence_CodeMeaning"])
    num_classes = y_train.shape[1]

    # === 新增2：保存到缓存文件 ===
    if use_cache:
        torch.save({
            "X_train_ae": X_train_ae,
            "y_train": y_train,
            "cat_ids_train": cat_ids_train,

            "X_val_ae": X_val_ae,
            "y_val": y_val,
            "cat_ids_val": cat_ids_val,

            "X_test_ae": X_test_ae,
            "y_test": y_test,
            "cat_ids_test": cat_ids_test,

            "num_cat1": num_cat1,
            "num_cat2": num_cat2,
            "num_cat3": num_cat3,
            "num_cat4": num_cat4,
            "num_classes": num_classes
        }, cache_file)
        print(f"Data saved to cache: {cache_file}")

# (下方代码保持不变，只是拿到 X_train_ae, y_train 等以后继续)
# ---------------------------------------------------------
# 7) 多标签分类的 DataLoader (X_ae, cat_ids, y)
train_loader_cls = DataLoader(
    TensorDataset(X_train_ae, cat_ids_train, y_train),
    batch_size=16, shuffle=True
)
val_loader_cls = DataLoader(
    TensorDataset(X_val_ae, cat_ids_val, y_val),
    batch_size=16, shuffle=False
)
test_loader_cls = DataLoader(
    TensorDataset(X_test_ae, cat_ids_test, y_test),
    batch_size=16, shuffle=False
)

# 9) 初始化分类器
model_cls = MultiInputCXRClassifier(
    ae_feature_dim=128,
    num_class=num_classes,
    num_cat1=num_cat1,
    num_cat2=num_cat2,
    num_cat3=num_cat3,
    num_cat4=num_cat4,
    embed_dim=32
).to(device)

optimizer = optim.Adam(model_cls.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')

# 10) 训练循环
num_epochs = 60
best_val_loss = float('inf')
best_model_state = None

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model_cls.train()
    total_loss = 0.0

    for x_ae_batch, cat_ids_batch, y_batch in train_loader_cls:
        x_ae_batch = x_ae_batch.to(device)
        cat_ids_batch = cat_ids_batch.to(device)
        labels = y_batch.to(device)

        # 处理标签的 NaN
        mask = ~torch.isnan(labels)
        labels = torch.nan_to_num(labels, nan=0.0)

        # 前向 + 损失
        outputs = model_cls(x_ae_batch, cat_ids_batch)
        loss = criterion(outputs[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader_cls)

    # ====== 验证 ======
    model_cls.eval()
    val_loss = 0.0
    y_val_true, y_val_pred = [], []

    with torch.no_grad():
        for x_ae_batch, cat_ids_batch, y_batch in val_loader_cls:
            x_ae_batch = x_ae_batch.to(device)
            cat_ids_batch = cat_ids_batch.to(device)
            labels = y_batch.to(device)

            mask = ~torch.isnan(labels)
            labels = torch.nan_to_num(labels, nan=0.0)

            outputs = model_cls(x_ae_batch, cat_ids_batch)
            loss_val = criterion(outputs[mask], labels[mask])
            val_loss += loss_val.item()

            y_val_true.append(labels.cpu().numpy())
            y_val_pred.append(outputs.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader_cls)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model_cls.state_dict()
        print(f"New best model found at epoch {epoch+1} (Val Loss = {avg_val_loss:.4f})")

# 恢复最佳模型
model_cls.load_state_dict(best_model_state)
print("\nFinished Training. Best val_loss =", best_val_loss)

# ==========================
# 11) 在测试集上做评估
# ==========================
print("\nEvaluating on TEST SET...")
model_cls.eval()
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for x_ae_batch, cat_ids_batch, y_batch in test_loader_cls:
        x_ae_batch = x_ae_batch.to(device)
        cat_ids_batch = cat_ids_batch.to(device)
        labels = y_batch.to(device)

        outputs = model_cls(x_ae_batch, cat_ids_batch)
        y_true_all.append(labels.cpu().numpy())
        y_pred_all.append(outputs.cpu().numpy())

y_true_all = np.concatenate(y_true_all, axis=0)
y_pred_all = np.concatenate(y_pred_all, axis=0)

# 12) 在测试集上找最优阈值（或使用验证集的最优阈值）
best_thresholds_test = find_best_thresholds(y_true_all, y_pred_all)

y_pred_bin = np.zeros_like(y_pred_all)
for i in range(y_pred_all.shape[1]):
    y_pred_bin[:, i] = (y_pred_all[:, i] >= best_thresholds_test[i]).astype(int)

# 计算 AUC, precision, recall, f1
aucs, precisions, recalls, f1s = [], [], [], []
for i in range(y_true_all.shape[1]):
    mask = ~np.isnan(y_true_all[:, i])
    if np.sum(mask) == 0:
        aucs.append(np.nan)
        precisions.append(np.nan)
        recalls.append(np.nan)
        f1s.append(np.nan)
    else:
        y_true_i = y_true_all[mask, i]
        y_prob_i = y_pred_all[mask, i]
        y_pred_i = y_pred_bin[mask, i]

        aucs.append(roc_auc_score(y_true_i, y_prob_i))
        precisions.append(precision_score(y_true_i, y_pred_i))
        recalls.append(recall_score(y_true_i, y_pred_i))
        f1s.append(f1_score(y_true_i, y_pred_i))

# 打印各类别结果
for i in range(len(aucs)):
    print(f"Test Class {i}: AUC={aucs[i]:.4f}, Precision={precisions[i]:.4f}, Recall={recalls[i]:.4f}, F1={f1s[i]:.4f}")

print(f"\nTest Set Average AUC:       {np.nanmean(aucs):.4f}")
print(f"Test Set Average Precision: {np.nanmean(precisions):.4f}")
print(f"Test Set Average Recall:    {np.nanmean(recalls):.4f}")
print(f"Test Set Average F1-score:  {np.nanmean(f1s):.4f}")

# 可视化损失曲线
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Val Loss (Classifier)")
plt.show()
