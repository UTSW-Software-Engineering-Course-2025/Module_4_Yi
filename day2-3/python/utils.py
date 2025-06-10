# utils.py
import numpy as np
import matplotlib.pyplot as plt

def jordan_decomp(ca, return_mask=False):
    """
    返回:
      yP        -- 只包含正增量帧 (长度 = mask_pos.sum())
      mask_pos  -- 布尔掩码, 与 ca 等长
    """
    diff        = np.diff(np.insert(ca, 0, ca[0]))
    mask_pos    = diff > 0                       # True at positive increments
    pos_diff    = np.where(mask_pos, diff, 0.0)

    y_full      = np.cumsum(pos_diff)            # f+ 全序列
    yP          = y_full[mask_pos]               # 仅正增量帧

    if return_mask:
        return yP, mask_pos
    return yP

def plot_activity_map(actmap, out_png):
    q0,q1 = np.quantile(actmap,[0.005,0.995])
    plt.figure(figsize=(12,4))
    plt.imshow(actmap, aspect='auto', vmin=q0, vmax=q1, cmap='jet')
    plt.colorbar(); plt.title("Calcium activity map")
    plt.savefig(out_png, dpi=300); plt.close()

def logsumexp(a, axis=None, keepdims=False):
    """NumPy 版 logsumexp"""
    a_max = np.max(a, axis=axis, keepdims=True)
    res   = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return res if keepdims else np.squeeze(res, axis=axis)

def plot_binary_map(binmap, fname, vmax=1):
    """绘制二值 spike map，binmap: [neurons × frames] 0/1"""
    plt.figure(figsize=(12,4))
    plt.imshow(binmap, aspect='auto', cmap='gray', vmin=0, vmax=vmax)
    plt.colorbar(); plt.title("Binary spike map")
    plt.savefig(fname, dpi=300); plt.close()
