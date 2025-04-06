import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MNISTData import create_dataloaders
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
import copy

class ViTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model)
        )
    
    def forward(self, x: torch.Tensor):
        xnorm = self.norm1.forward(x)
        xenc = self.mha.forward(xnorm, xnorm, xnorm, need_weights=False)[0]
        x = xenc + x
        xnorm = self.norm2.forward(x)
        x: torch.Tensor = self.fc(xnorm) + x
        return x
    
class ViTEncoder(nn.Module):
    def __init__(self, layer: ViTEncoderLayer, num_layers:int):
        super().__init__()
        self.layers = nn.Sequential(
            *[copy.deepcopy(layer) for _ in range(num_layers)]
        )
    
    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim0, self.dim1)
    

class CLSToken(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.CLS = nn.Parameter(torch.randn(1, 1, d_model))  # 1トークンぶん

    def forward(self, x: torch.Tensor):
        B = x.size(0)  # バッチサイズに合わせて拡張
        cls_token = self.CLS.expand(B, -1, -1)  # [1,1,D] -> [B,1,D]
        return torch.cat([cls_token, x], dim=1)  # 先頭にCLSを追加

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.PE = nn.Parameter(torch.randn(1, max_len, d_model))  # 最大長に対応

    def forward(self, x: torch.Tensor):
        B, T, D = x.size()
        if T > self.max_len:
            raise ValueError(f"入力長 {T} は max_len {self.max_len} を超えています。")

        return x + self.PE[:, :T, :]  # 必要な長さだけ取り出して加算

class Select(nn.Module):
    def __init__(self, i: int, dim: int, keepdim: bool = False):
        super().__init__()
        self.i = i
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keepdim:
            # index_select + unsqueezeで次元を保持
            return x.index_select(self.dim, torch.tensor([self.i], device=x.device))
        else:
            return x.select(self.dim, self.i)

class SimpleViT(nn.Module):
    def __init__(self, img_size: int=28, patch_size: int=4, embeddim:int=32, outdim: int=10, nhead: int=4, num_layers: int=3):
        super().__init__()
        patches = img_size // patch_size
        self.vit = nn.Sequential(
            nn.Conv2d(1, embeddim, patch_size, patch_size),
            nn.Flatten(-2),
            Transpose(-2,-1),
            CLSToken(embeddim),
            PositionalEncoding(patches**2+1, embeddim),
            ViTEncoder(ViTEncoderLayer(
                d_model=embeddim, nhead=nhead
            ), num_layers=num_layers),
            Select(0, -2)
        )
        self.outproj = nn.Sequential(
            nn.Linear(embeddim, embeddim*2),
            nn.ReLU(),
            nn.Linear(embeddim*2, outdim)
        )
        self.__init_weights__()

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.vit.forward(x.unsqueeze(1))
        x: torch.Tensor = self.outproj(x)
        return x

def train1epoch(model: SimpleViT, traindataloader: DataLoader, optimizer, criterion, device='cuda'):
    """Lossを返す"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in traindataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 勾配をゼロにリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model.forward(inputs)
        
        # 損失計算
        loss = criterion(outputs, labels)
        
        # 逆伝播とパラメータ更新
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(traindataloader.dataset)
    return epoch_loss

def validate(model: SimpleViT, validatedataloader: DataLoader, criterion, device='cuda'):
    """検証用データセットでの評価、F1スコアも計算する"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in validatedataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(validatedataloader.dataset)
    val_acc = correct / total
    # F1スコアの計算（マクロ平均）
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1

def train(model: SimpleViT, traindataloader: DataLoader, validatedataloader: DataLoader, 
          epochs: int, device='cuda', lr=0.001):
    """Train 1 Epoch * Epoch, Loss, Accuracy, F1スコアの遷移を返す。検証F1が最高のモデルを保存する"""
    model = model.to(device)
    
    # 最適化関数とloss関数の設定
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    best_f1 = -1.0
    best_model_state = None
    
    for epoch in range(epochs):
        # 訓練
        train_loss = train1epoch(model, traindataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 検証
        val_loss, val_acc, val_f1 = validate(model, validatedataloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}')
        
        # 現在のF1スコアがこれまでの最高より良ければモデルを保存
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s
    }
    return history, best_model_state

def main(trainpath: str, testpath: str, outpath: str, modelpath: str, epochs=10, batch_size=64):
    """trainpath, testpathの存在確認, outpath作成, modelpathの存在を確認して存在すれば読み込み, 基本はoutpath内に出力する"""
    train_loader, val_loader, test_loader = create_dataloaders(
        trainpath=trainpath,
        testpath=testpath,
        splitrate=0.8,
        batch_size=batch_size
    )
    
    # 出力ディレクトリの作成
    os.makedirs(outpath, exist_ok=True)
    from helpers import get_device, visualize_misclassified
    # デバイスの設定
    device = get_device()
    print(f"Using device: {device}")
    
    # モデルの初期化
    model = SimpleViT()
    
    # モデルの読み込み（もし存在すれば）
    if os.path.exists(modelpath):
        print(f"モデルを読み込み中: {modelpath}")
        model.load_state_dict(torch.load(modelpath))
    
    # モデルのトレーニング（historyと最も信頼性の高いモデルの状態を取得）
    history, best_model_state = train(model, train_loader, val_loader, epochs=epochs, device=device)
    
    # 最も信頼性の高いモデルの保存
    best_model_path = os.path.join(outpath, 'mnist_best_model.pth')
    torch.save(best_model_state, best_model_path)
    print(f'最も信頼性の高いモデルを保存しました: {best_model_path}')
    
    # 学習曲線のプロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accs'], label='Validation Accuracy')
    plt.plot(history['val_f1s'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Accuracy & F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'learning_curves.png'))
    
    visualize_misclassified(model, dataloader=val_loader, device=device, output_path=os.path.join(outpath, 'visualize_misclassified.png'), max_images=20)

    # テストデータでの予測
    model.eval()
    model = model.to(device)
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            # テストデータはラベルがないので、inputsはタプルではなくテンソル
            inputs = inputs.to(device)
            outputs = model.forward(inputs)
            
            # 予測クラスを取得
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # 予測結果をDataFrameに変換
    results_df = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    # CSVファイルとして保存
    results_path = os.path.join(outpath, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'予測結果を保存しました: {results_path}')
    
    return history

if __name__ == "__main__":
    from helpers import currents
    dirs = currents(
            [
                "data/train.csv",
                "data/test.csv",
                "outViT/",
                "outViT/mnist_best_model.pth"
            ]
        )
    print(dirs)
    main(
        *dirs,
        epochs=60,
        batch_size=64
    )
