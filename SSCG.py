import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
# 定義 通道自注意力 (Channel Self-Attention) 模組
'''
# 定義 Self-Attention 取代 CBAM 的 Self-Attention + BiGRU 模型
class SelfAttention_CBAM_BiGRU(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, num_heads=10, hidden_dim=64, dropout=0.5):
        super(SelfAttention_CBAM_BiGRU, self).__init__()
        # 空間自注意力 (Spatial Self-Attention)
        self.spatial_attention = SpatialSelfAttention(input_dim=input_dim, num_heads=num_heads)
        # 通道自注意力 (Channel Self-Attention)
        self.channel_attention = ChannelSelfAttention(seq_len=seq_len, num_heads=num_heads)
        print(f"SpatialSelfAttention - num_classes: {num_classes}")

        # BiGRU 層
        self.bi_gru_1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bi_gru_2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # 全局池化層和全連接層
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # 通道自注意力
        x = self.channel_attention(x)
        
        # 空間自注意力
        x = self.spatial_attention(x)

        # BiGRU 層
        gru_output, _ = self.bi_gru_1(x)
        gru_output, _ = self.bi_gru_2(gru_output)

        # 全局池化和全連接層
        gru_output = gru_output.permute(0, 2, 1)  # (batch, hidden_dim*2, seq_len)
        pooled_output = self.global_pooling(gru_output).squeeze(-1)  # (batch, hidden_dim*2)

        dense_output = self.dense1(pooled_output)  # (batch, 64)
        dense_output = torch.relu(dense_output)
        dense_output = self.dropout(dense_output)
        output = self.output_layer(dense_output)  # (batch, num_classes)

        return output
'''
# 定義 通道自注意力 (Channel Self-Attention) 模組

class ChannelSelfAttention(nn.Module):
    def __init__(self, seq_len, num_heads=10):
        super(ChannelSelfAttention, self).__init__()
        print(f"ChannelSelfAttention - embed_dim: {seq_len}, num_heads: {num_heads}")
        # 因為我們需要對 seq_len 進行注意力，所以 embed_dim 設為 seq_len
        self.multihead_attention = nn.MultiheadAttention(embed_dim=seq_len, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(seq_len)

    def forward(self, x):
        # x: (batch, seq_len, channels=10)
        # 我們希望對 channels 進行注意力，因此將 channels 視為序列
        # 轉置為 (batch, channels=10, seq_len=90)
        x_permuted = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        # 再轉置為 (batch, seq_len=10, embed_dim=90)  # 這裡 embed_dim 應為 input_dim=10
        #x_permuted = x_permuted.permute(0, 2, 1)  # (batch, seq_len=90, embed_dim=10)
        # 這樣 MultiheadAttention 的 embed_dim=10 與最後一維匹配
        attn_output, _ = self.multihead_attention(x_permuted, x_permuted, x_permuted)
        attn_output = self.layer_norm(attn_output + x_permuted)  # 殘差連接並正則化
        return attn_output.permute(0, 2, 1)  # (batch, seq_len=90, embed_dim=10)

# 定義 空間自注意力 (Spatial Self-Attention) 模組
class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=10):
        super(SpatialSelfAttention, self).__init__()
        print(f"SpatialSelfAttention - embed_dim: {input_dim}, num_heads: {num_heads}")
        # embed_dim 設為 input_dim=10
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch, seq_len=90, channels=10)
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = self.layer_norm(attn_output + x)  # 殘差連接並正則化
        return attn_output  # (batch, seq_len=90, channels=10)

# 定義 Self-Attention + BiGRU 模型，並在輸入處加入 Batch Normalization
class SelfAttention_CBAM_BiGRU2(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, num_heads=5, hidden_dim=64, dropout=0.5):
        super(SelfAttention_CBAM_BiGRU2, self).__init__()

        # Batch Normalization (應用於 seq_len)
        self.seq_batch_norm = nn.BatchNorm1d(seq_len)

        # 通道自注意力 (Channel Self-Attention)
        self.channel_attention = ChannelSelfAttention(seq_len=seq_len, num_heads=10)
        # 空間自注意力 (Spatial Self-Attention)
        self.spatial_attention = SpatialSelfAttention(input_dim=input_dim, num_heads=10)

        # BiGRU 層
        self.bi_gru_1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bi_gru_2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # 全局池化層和全連接層
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        
        # 將 batch 和通道調整到第 2 和第 3 維度，使 BatchNorm1d 對 seq_len 進行正則化
        #x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)

        # 對序列長度 (seq_len) 進行 Batch Normalization
        x = self.seq_batch_norm(x)

        # 將形狀調整回原來的形狀 (batch_size, seq_len, input_dim)
        #x = x.permute(0, 2, 1)

        # 通道自注意力
        x = self.channel_attention(x)

        # 空間自注意力
        x = self.spatial_attention(x)

        # BiGRU 層
        #print(f"Input to GRU1 shape: {x.shape}")
        gru_output, _ = self.bi_gru_1(x)
        #print(f"Output from GRU1 shape: {gru_output.shape}")
        gru_output, _ = self.bi_gru_2(gru_output)  # (batch, seq_len, hidden_dim*2)
        #print(f"Output from GRU2 shape: {gru_output.shape}")  # 印出第二層 GRU 的輸出形狀
        # 全局池化和全連接層
        gru_output = gru_output.permute(0, 2, 1)  # (batch, hidden_dim*2, seq_len)
        pooled_output = self.global_pooling(gru_output).squeeze(-1)  # (batch, hidden_dim*2)

        dense_output = self.dense1(pooled_output)  # (batch, 64)
        dense_output = torch.relu(dense_output)
        dense_output = self.dropout(dense_output)
        output = self.output_layer(dense_output)  # (batch, num_classes)

        return output


class SSCGTrainer2:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(1)  # 設定使用編號 0 的 GPU
    def visualize_model(self, model, input_tensor):
        output = model(input_tensor)
        return make_dot(output, params=dict(list(model.named_parameters())))

    def sliding_window(self, data, labels, window_size):
        X = []
        y = []
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            window_labels = labels[i:i + window_size]
            # 確保 mode 結果不為 NaN
            mode_result = pd.Series(window_labels).mode()
            if not mode_result.empty and pd.notna(mode_result[0]):
                most_frequent_label = mode_result[0]
            else:
                continue  # 如果 mode 是 NaN 或空的，則跳過這個窗口
            X.append(window_data)
            y.append(most_frequent_label)
        return np.array(X), np.array(y)

    def train_model(self, train_file_path, selected_features=None, epochs=20, batch_size=64, window_size=120):
        train_data = pd.read_csv(train_file_path)

        if selected_features is None:
            selected_features = [col for col in train_data.columns if col not in ['Action', 'Absolute Time']]

        X = train_data[selected_features].values
        y = train_data['Action'].values

        # 先篩選出非 NaN 的標籤，並進行編碼
        label_encoder = LabelEncoder()
        non_nan_mask = ~pd.isna(y)
        y_non_nan = y[non_nan_mask]
        y_encoded_non_nan = label_encoder.fit_transform(y_non_nan)

        # 創建一個編碼後的 y，將 NaN 值保留
        y_encoded = np.full_like(y, np.nan, dtype=float)
        y_encoded[non_nan_mask] = y_encoded_non_nan

        print("Classes:", label_encoder.classes_)

        X_windows, y_windows = self.sliding_window(X, y_encoded, window_size)
        print(f"滑動窗口後的特徵形狀: {X_windows.shape}")
        print(f"滑動窗口後的標籤形狀: {y_windows.shape}")

        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        y_tensor = torch.tensor(y_windows, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_windows.shape[2]
        num_classes = len(label_encoder.classes_)

        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")

        model = SelfAttention_CBAM_BiGRU2(input_dim=input_dim, num_classes=num_classes, seq_len=window_size)
        model.to(self.device)
        print(f"模型已移至設備: {self.device}")
        print('Using CUDA device:', self.device)
        # 可視化模型
        example_input = torch.randn(1, window_size, input_dim).to(self.device)  # 範例輸入 (batch, seq_len, input_dim)
        dot = self.visualize_model(model, example_input)
        dot.render("model_architectureSSCG", format="png")  # 將圖儲存為 PNG
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        if self.model_path:
            torch.save(model.state_dict(), self.model_path)
        else:
            print("Model not saved. Provide a valid model path.")

        self.plot_metrics(epoch_losses, epoch_accuracies)
        return model

    def plot_metrics(self, losses, accuracies):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses, label="Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label="Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
