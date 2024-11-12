import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from collections import Counter
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
class SelfAttention_CBAM_BiGRU(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, num_heads=5, hidden_dim=64, dropout=0.5):
        super(SelfAttention_CBAM_BiGRU, self).__init__()

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
        gru_output, _ = self.bi_gru_1(x)
        gru_output, _ = self.bi_gru_2(gru_output)  # (batch, seq_len, hidden_dim*2)

        # 全局池化和全連接層
        gru_output = gru_output.permute(0, 2, 1)  # (batch, hidden_dim*2, seq_len)
        pooled_output = self.global_pooling(gru_output).squeeze(-1)  # (batch, hidden_dim*2)

        dense_output = self.dense1(pooled_output)  # (batch, 64)
        dense_output = torch.relu(dense_output)
        dense_output = self.dropout(dense_output)
        output = self.output_layer(dense_output)  # (batch, num_classes)

        return output

class SSCGTester2:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _mode_of_data(self, window):
        count = Counter(window)
        return count.most_common(1)[0][0]

    def filter_actions(self, dataset, frequency):
        filtered_data = []
        previous_action = None

        for i, action in enumerate(dataset):
            if action != previous_action:
                window = dataset[i:i + (2 * frequency)]
                if self._mode_of_data(window) == action:
                    pass
                else:
                    if self._mode_of_data(window + dataset[i - frequency:i + frequency]) != action:
                        action = self._mode_of_data(window)
                    else:
                        action = previous_action
            filtered_data.append(action)
            previous_action = action

        return filtered_data

    def plot_results(self, result_df, overall_accuracy, cm, labels):
        actions = result_df['Action'].tolist()
        accuracies = result_df['Accuracy'].tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        palette = sns.color_palette("husl", len(actions))
        sns.barplot(x='Action', y='Accuracy', data=result_df, ax=ax1, palette=palette)
        ax1.axhline(overall_accuracy, color='red', linestyle='--')
        ax1.text(len(actions) - 0.5, overall_accuracy + 1, f'Accuracy (Total): {overall_accuracy:.2f}', color='black', ha='right', fontsize=16)
        ax1.set_title('Accuracy Analysis of Different Actions', fontsize=20, pad=12)
        ax1.set_xlabel('Action', fontsize=20, labelpad=10)
        ax1.set_ylabel('Accuracy (%)', fontsize=20, labelpad=10)
        ax1.set_ylim(0, 100)
        plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap=plt.cm.Blues, ax=ax2)
        ax2.set_title('Confusion Matrix', fontsize=20, pad=12)
        plt.tight_layout()
        plt.show()

    def test_model(self, test_file_path, selected_features=None, frequency=6, save_predictions_path=None, window_size=90, save_detailed_csv_path=None):
        test_data = pd.read_csv(test_file_path)
        if selected_features is None:
            selected_features = [col for col in test_data.columns if col not in ['Action', 'Absolute Time']]
        X_test = test_data[selected_features].values
        y_test = test_data['Action'].values

        X_windows = []
        y_first_labels = []

        for i in range(len(X_test) - window_size + 1):
            window = X_test[i:i + window_size]
            if pd.isna(y_test[i]):
                continue

            # 檢查後續 window_size 筆資料中的最常見標籤
            next_window = y_test[i:i + window_size]
            if self._mode_of_data(next_window) != y_test[i]:
                continue  # 若當前標籤與後續 window_size 筆資料中最常見標籤不同，跳過此資料

            X_windows.append(window)
            y_first_labels.append(y_test[i])
        # One-hot 編碼標籤
        y_test_encoded = pd.get_dummies(y_first_labels).values
        # 將滑動窗口的特徵轉換為 PyTorch 張量
        X_test_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32).to(self.device)
        # 創建 DataLoader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        # 載入訓練好的模型，並加入 seq_len
        model = SelfAttention_CBAM_BiGRU(input_dim=X_test.shape[1], num_classes=y_test_encoded.shape[1], seq_len=window_size)
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_true, axis=1)

        cm = confusion_matrix(y_test_classes, y_pred_classes)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        predictions_df = pd.DataFrame({
            'True': pd.get_dummies(y_first_labels).columns[y_test_classes],
            'Predict': pd.get_dummies(y_first_labels).columns[y_pred_classes]
        })
        predictions_df['Filter_Predict'] = self.filter_actions(predictions_df['Predict'], frequency)

        overall_accuracy = np.mean(predictions_df['True'] == predictions_df['Filter_Predict']) * 100

        result_df = pd.DataFrame({
            'Action': pd.get_dummies(y_first_labels).columns,
            'Accuracy': class_accuracy * 100
        })
        labels = pd.get_dummies(y_first_labels).columns
        self.plot_results(result_df, overall_accuracy, cm, labels)

        print(result_df)
        print(f"\nTotal accuracy: {overall_accuracy:.2f}%")

        if save_predictions_path:
            predictions_df = pd.concat([predictions_df, test_data[selected_features].iloc[:len(predictions_df)].reset_index(drop=True)], axis=1)
            predictions_df.to_csv(save_predictions_path, index=False)
            print(f"Filtered predictions saved to {save_predictions_path}")
                # 保存詳細的預測結果到 CSV（包含選定的特徵、Action、Window 標籤和預測結果）
        if save_detailed_csv_path:
            detailed_data = []
            for window_id, (window, true_label, pred_label) in enumerate(zip(X_windows, y_first_labels, predictions_df['Filter_Predict'])):
                row_data = {
                    'Window_ID': window_id,
                    'True_Label': true_label,
                    'Predicted_Label': pred_label,
                }
                for feature_idx, feature_value in enumerate(window[0]):  # 取窗口內的第一筆資料的特徵
                    row_data[selected_features[feature_idx]] = feature_value
                detailed_data.append(row_data)

            detailed_results_df = pd.DataFrame(detailed_data)
            detailed_results_df.to_csv(save_detailed_csv_path, index=False)
            print(f"Detailed predictions saved to {save_detailed_csv_path}")
