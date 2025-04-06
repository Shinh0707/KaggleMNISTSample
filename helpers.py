def get_caller_file_path(depth=1):
    import inspect
    import os
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is not None:
            frame = frame.f_back
    if frame is not None:
        return os.path.abspath(frame.f_code.co_filename)
    return None

def current(path: str):
    import os
    return os.path.join(os.path.dirname(get_caller_file_path(2)), path)

def currents(paths: list[str]):
    import os
    dir = os.path.dirname(get_caller_file_path(2))
    return [os.path.join(dir, path) for path in paths]

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

def visualize_misclassified(model, dataloader, device, output_path: str, max_images=20):
    """
    モデルの検証データにおいて、誤分類された画像を以下の情報とともに表示します:
      - 正解ラベル
      - 予測ラベル
      - 予測出力（softmax後）のうち、正解ラベルが何番目に大きいか（順位）
    誤分類された画像は、正解ラベルの予測順位が悪い順（大きい値）にソートされ、
    最大 max_images 件をグリッド状に表示します。
    """
    import torch.nn.functional as F
    import numpy as np
    import math
    import torch
    import matplotlib.pyplot as plt
    model.eval()
    misclassified = []  # (画像, 正解ラベル, 予測ラベル, 正解ラベルの順位) のタプルリスト
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # モデルの出力がタプルの場合は最初の要素を使用（例: SimpleAE）
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # softmaxで確率に変換
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                # 正解と予測が異なる場合
                if true_label != pred_label:
                    prob_values = probs[i].cpu().numpy()
                    # 降順に並べたインデックス
                    sorted_indices = np.argsort(-prob_values)
                    # 正解ラベルの順位（1から始まる）
                    rank = np.where(sorted_indices == true_label)[0][0] + 1
                    misclassified.append((inputs[i].cpu(), true_label, pred_label, rank))
    
    # 正解ラベルの順位が大きい（＝悪い順位）の順にソート
    misclassified.sort(key=lambda x: x[3], reverse=True)
    misclassified = misclassified[:max_images]
    
    n_images = len(misclassified)
    if n_images == 0:
        print("検証データに誤分類された画像はありませんでした。")
        return
    
    # グリッド表示
    ncols = int(math.sqrt(n_images))
    nrows = math.ceil(n_images / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    # 1次元配列に変換（axesが1次元の場合も含む）
    if nrows * ncols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()
    
    # すべての軸の枠線を消す
    for ax in axes:
        ax.axis('off')
    
    for i, (img, true_label, pred_label, rank) in enumerate(misclassified):
        ax = axes[i]
        # 画像の次元が [1, H, W] の場合は squeeze して [H, W] に
        image_np = img.squeeze().numpy()
        ax.imshow(image_np, cmap='gray')
        ax.set_title(f"Currect: {true_label}\nPred: {pred_label}\nRank: {rank}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)