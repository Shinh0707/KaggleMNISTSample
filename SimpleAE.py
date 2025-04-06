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

class SimpleEncoder(nn.Module):
    def __init__(self, embeddim:int=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(32, embeddim*2),
            nn.ReLU(),
            nn.Linear(embeddim*2, embeddim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SimpleDecoder(nn.Module):
    def __init__(self, embeddim:int=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(embeddim, embeddim * 2),
            nn.ReLU(),
            nn.Linear(embeddim * 2, 32),
            nn.Unflatten(1, torch.Size([32, 1, 1])),
            nn.UpsamplingBilinear2d(scale_factor=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1),
            nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(2, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SimpleAutoEncoder(nn.Module):
    def __init__(self, embeddim: int=16):
        super().__init__()
        self.encoder = SimpleEncoder(embeddim=embeddim)
        self.decoder = SimpleDecoder(embeddim=embeddim)
        self.loss = nn.MSELoss()
        self.__init_weights__()

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # He初期化: 分散 = 2/fan_in (ReLU 用)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 多くの Linear 層は ReLU を用いるので He 初期化を採用
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,28,28] -> [B,1,28,28]
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return encoded.detach(), self.loss.forward(decoded, x.detach())
    
    def encode(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # [B,28,28] -> [B,1,28,28]
        return self.encoder.forward(x).detach()
    
    def decode(self, x: torch.Tensor):
        return self.decoder.forward(x).detach()

class SimpleAutoEncoderPredictor(nn.Module):
    def __init__(self, embeddim: int=16):
        super().__init__()
        self.ae = SimpleAutoEncoder(embeddim=embeddim)
        self.outproj = nn.Sequential(
            nn.Linear(embeddim, embeddim*2),
            nn.ReLU(),
            nn.Linear(embeddim*2, embeddim)
        )

    def forward(self, x: torch.Tensor):
        x, loss = self.ae.forward(x)
        x: torch.Tensor = self.outproj(x)
        return x, loss

def train1epoch(model: SimpleAutoEncoderPredictor, traindataloader: DataLoader, optimizer, criterion, device='cuda'):
    """Lossを返す"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in traindataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 勾配をゼロにリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs, loss = model.forward(inputs)
        
        # 損失計算
        loss = loss + criterion(outputs, labels)
        
        # 逆伝播とパラメータ更新
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(traindataloader.dataset)
    return epoch_loss

def validate(model: SimpleAutoEncoderPredictor, validatedataloader: DataLoader, criterion, device='cuda'):
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
            
            outputs, loss = model.forward(inputs)
            loss = loss + criterion(outputs, labels)
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

def train(model: SimpleAutoEncoderPredictor, traindataloader: DataLoader, validatedataloader: DataLoader, 
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
    model = SimpleAutoEncoderPredictor()
    
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
            outputs, _ = model.forward(inputs)
            
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
                "outAE/",
                "outAE/mnist_best_model.pth"
            ]
        )
    print(dirs)
    main(
        *dirs,
        epochs=60,
        batch_size=64
    )
