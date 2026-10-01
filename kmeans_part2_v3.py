"""HW1 Part 2 v1：排除白色背景，加入角色的 HSV 色彩直方圖。

使用：把本檔、features.npy、pal_images/ 放在同一資料夾。
執行：python kmeans_part2_v1.py
輸出：與本程式同名的 CSV；上傳 Kaggle 取得 ARI。

只用 NumPy、Pillow 與 Python 標準函式庫。
沒有查詢角色屬性、人工標籤或額外的特徵抽取模型。
本版不讀 metadata.json，讓這次實驗集中在圖片特徵。
Kaggle 基準 ARI = 0.19691；此版本的 ARI 尚待上傳驗證。
"""
from pathlib import Path
from collections import deque
import csv
import numpy as np
from PIL import Image

K = 7
SEED = 42
N_INIT = 20
MAX_ITER = 300
COLOR_WEIGHT = 1      # 新增圖片特徵的權重；先保持 0.50
IMAGE_SIZE = 256         # 縮圖上限，加速顏色統計
WHITE_THRESHOLD = 245   # R、G、B 都 >= 245 才算近白色
HUE_BINS = 24


def foreground_mask(rgb):
    """排除與圖片邊界連通的近白色區域。

    以邊界為起點向上下左右擴張，只穿越近白色像素。
    被有色輪廓包圍的白色部位仍可保留；這是近似的背景排除，
    若角色的白色部位直接連到背景，仍可能被排除。
    """
    near_white = np.all(rgb >= WHITE_THRESHOLD, axis=2)
    height, width = near_white.shape
    background = np.zeros((height, width), dtype=bool)
    queue = deque()

    def add(y, x):
        if near_white[y, x] and not background[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(width):
        add(0, x)
        add(height - 1, x)
    for y in range(height):
        add(y, 0)
        add(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            add(y - 1, x)
        if y + 1 < height:
            add(y + 1, x)
        if x > 0:
            add(y, x - 1)
        if x + 1 < width:
            add(y, x + 1)
    return ~background


def image_features(path):
    """把一張圖片轉成 148 個顏色特徵，以及前景像素比例。"""
    with Image.open(path) as source:
        image = source.convert('RGB')
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
        mask = foreground_mask(rgb)
        if mask.sum() < 10:
            raise ValueError(f'{path.name} 找不到足夠前景像素。')
        hsv = np.asarray(image.convert('HSV'), dtype=np.float64)[mask] / 255.0

    # H 是色相（什麼顏色），S 是飽和度（鮮豔程度），V 是亮度。
    h, s, v = hsv.T
    chromatic = s >= 0.12
    # 將彩色像素分成 24 種色相、2 種飽和度、3 種亮度。
    hue = np.minimum((h[chromatic] * HUE_BINS).astype(int), HUE_BINS - 1)
    saturation = np.searchsorted([0.55], s[chromatic])
    value = np.searchsorted([0.30, 0.70], v[chromatic])
    indices = (hue * 2 + saturation) * 3 + value
    color = np.bincount(indices, minlength=HUE_BINS * 2 * 3)
    color = color.astype(np.float64).reshape(HUE_BINS, 2, 3)

    # 相鄰色相共享部分統計；首尾連通，避免紅色落在色相兩端而被拆開。
    color = (0.50 * color + 0.25 * np.roll(color, 1, axis=0)
             + 0.25 * np.roll(color, -1, axis=0))

    # 低飽和度像素不靠色相分類，另外統計黑、灰、白的亮度分布。
    gray_bins = np.searchsorted([0.25, 0.50, 0.75], v[~chromatic])
    gray = np.bincount(gray_bins, minlength=4).astype(np.float64)
    histogram = np.concatenate([color.ravel(), gray])
    histogram /= histogram.sum()
    # 平方根轉換，減少最大顏色區塊完全主導距離的情況。
    return np.sqrt(histogram), float(mask.mean())


def normalize_block(block):
    """每欄置中，再用整組特徵的共同尺度縮放。

    使不同維度數的特徵組有可比較的整體尺度。
    不把稀少顏色的每一欄各自放大到相同變異數。
    """
    centered = block - block.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if scale < 1e-12:
        raise ValueError('特徵組沒有足夠變化，無法正規化。')
    return centered / scale


def squared_distances(data, centers):
    """回傳每筆資料到每個中心的平方距離，形狀為 (資料數, 群數)。"""
    # 第一軸是資料，第二軸是中心，第三軸是特徵。
    difference = data[:, None, :] - centers[None, :, :]
    return np.sum(difference ** 2, axis=2)


def kmeans_once(data, k, rng):
    """完成一次 K-means，回傳分群編號及群內平方距離總和。"""
    # 1. 隨機選 k 筆不同的資料，作為初始中心。
    indices = rng.choice(len(data), size=k, replace=False)
    centers = data[indices].copy()
    previous_labels = None

    for _ in range(MAX_ITER):
        # 2. 將每筆資料分配給最近的中心。
        distances = squared_distances(data, centers)
        labels = np.argmin(distances, axis=1)

        # 若分組已經不再改變，而且沒有空群，就停止。
        if (previous_labels is not None
                and np.array_equal(labels, previous_labels)
                and len(np.unique(labels)) == k):
            break

        # 若有空群，以離目前所屬中心最遠的資料重新設定中心。
        farthest = np.argsort(np.min(distances, axis=1))[::-1]
        empty_count = 0

        # 3. 對每群的成員逐欄取平均，更新中心。
        for group in range(k):
            members = data[labels == group]
            if len(members) > 0:
                centers[group] = members.mean(axis=0)
            else:
                centers[group] = data[farthest[empty_count]]
                empty_count += 1

        previous_labels = labels.copy()

    # 依最後的中心再計算一次分組及 K-means 的目標值。
    distances = squared_distances(data, centers)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(distances[np.arange(len(data)), labels]))
    return labels, inertia


def main():
    folder = Path(__file__).resolve().parent
    data = np.load(folder / 'features.npy', allow_pickle=False).astype(np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('features.npy 必須是有限數值的二維陣列。')
    if not 2 <= K < len(data) or not 0 <= COLOR_WEIGHT <= 1:
        raise ValueError('請檢查 K 與 COLOR_WEIGHT 設定。')

    images = folder / 'pal_images'
    # 使用整數 ID 查找，避免字串排序造成 1、10、100、2 的錯配。
    missing = [f'{i}.png' for i in range(len(data)) if not (images / f'{i}.png').is_file()]
    if missing:
        raise FileNotFoundError(
            'pal_images 資料夾內缺少圖片，例如：' + ', '.join(missing[:5])
            + '。請確認沒有多一層 pal_images/pal_images。')

    color_rows = []
    ratios = []
    print(f'開始讀取 {len(data)} 張圖片，請稍候...', flush=True)
    for pal_id in range(len(data)):
        row, ratio = image_features(images / f'{pal_id}.png')
        color_rows.append(row)
        ratios.append(ratio)
        if (pal_id + 1) % 50 == 0 or pal_id + 1 == len(data):
            print(f'圖片處理進度：{pal_id + 1}/{len(data)}', flush=True)
    colors = np.asarray(color_rows)

    # 乘以權重的平方根，因此合併後的平方距離為：
    # (1-w) * 原始特徵平方距離 + w * 圖片特徵平方距離。
    combined = np.concatenate([
        np.sqrt(1 - COLOR_WEIGHT) * normalize_block(data),
        np.sqrt(COLOR_WEIGHT) * normalize_block(colors),
    ], axis=1)
    print(f'原始特徵：{data.shape[1]} 維；新增顏色特徵：{colors.shape[1]} 維')
    print(f'保留像素比例範圍：{min(ratios):.1%} ~ {max(ratios):.1%}')
    print(f'新增顏色特徵權重：{COLOR_WEIGHT:.2f}；開始 K-means...', flush=True)

    rng = np.random.default_rng(SEED)
    best_labels = None
    best_inertia = float('inf')
    for _ in range(N_INIT):
        labels, inertia = kmeans_once(combined, K, rng)
        if len(np.unique(labels)) == K and inertia < best_inertia:
            best_labels = labels.copy()
            best_inertia = inertia
    if best_labels is None:
        raise ValueError('未找到具有 K 個非空群的結果。')

    output = Path(__file__).resolve().with_suffix('.csv')
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows((i, int(label)) for i, label in enumerate(best_labels))
    for group, count in enumerate(np.bincount(best_labels, minlength=K)):
        print(f'第 {group} 群：{count} 隻')
    print(f'已輸出：{output.name}')
    print('請上傳此 CSV 至 Kaggle 取得 ARI；本機不會顯示 ARI。')
    print('比較基準：Part 1 的 Kaggle ARI = 0.19691。')


if __name__ == '__main__':
    main()
