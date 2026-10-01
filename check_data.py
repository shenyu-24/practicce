from pathlib import Path
import numpy as np

# 找到這支程式所在的資料夾
folder = Path(__file__).resolve().parent

# 讀取助教提供的特徵資料
features = np.load(folder / "features.npy", allow_pickle=False)

# 顯示資料的基本資訊
print("Python 可以正常執行！")
print("NumPy 版本：", np.__version__)
print("資料形狀：", features.shape)
print("帕魯數量：", features.shape[0])
print("每隻帕魯的特徵數：", features.shape[1])
print("所有數值是否正常：", np.isfinite(features).all())