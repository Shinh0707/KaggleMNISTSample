import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

class MNISTDataset(Dataset):
    def __init__(self, filepath: str, onehot_labels: bool = False):
        super().__init__()
        self.onehot_labels = onehot_labels
        self.num_classes = 10  # MNISTは0-9の10クラス
        self.prepare(filepath)

    def prepare(self, filepath: str):
        df = pd.read_csv(filepath)
        # df['label']があれば、それとそれ以外を切り分ける
        if 'label' in df.columns:
            self.labels = torch.tensor(df['label'].values, dtype=torch.long)
            self.features = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32)
        else:
            self.labels = None
            self.features = torch.tensor(df.values, dtype=torch.float32)
        
        # 切り分けたものは28*28=784列のデータになっている
        # reshapeして[R,28,28]にする
        self.features = self.features.reshape(-1, 28, 28)
        # 値は0~255のintなので、255で割って0.0~1.0に正規化する
        self.features = self.features / 255.0
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            if self.onehot_labels:
                # one-hot encodingに変換
                onehot = F.one_hot(self.labels[idx], num_classes=self.num_classes).float()
                return self.features[idx], onehot
            else:
                return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]
        
def create_dataloaders(trainpath: str, testpath: str, splitrate: float=0.8, batch_size:int = 64):
    import os
    
     # パスの存在確認
    if not os.path.exists(trainpath):
        raise FileNotFoundError(f"トレーニングデータファイルが見つかりません: {trainpath}")
    
    if not os.path.exists(testpath):
        raise FileNotFoundError(f"テストデータファイルが見つかりません: {testpath}")
    
    # データセットのロード
    full_dataset = MNISTDataset(trainpath)
    test_dataset = MNISTDataset(testpath)
    
    # トレーニングデータを訓練用と検証用に分割
    train_size = int(splitrate * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader