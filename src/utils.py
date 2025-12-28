import numpy as np

def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm.astype(np.float32)
    if hm.max() <= 0:
        return hm
    hm = hm / hm.max()
    return hm