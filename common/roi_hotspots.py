# Imports
import os, cv2 # requires opencv-python
import numpy as np
from .utils import compose_filename

# region Helper functions
def robust_normalize(m):
    m = m.astype(np.float32)
    lo, hi = np.percentile(m, 2), np.percentile(m, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(m, dtype=np.float32)
    m = np.clip((m - lo) / (hi - lo), 0, 1)
    return m
#endregion

# ---------- Main function ----------
def roi_hotspots(
        image_path:str,
        create_edge_map: bool = True,
        create_local_variance_map: bool = True,
        create_high_freq_map: bool = True,
        save_hotspots_heat: bool = True,
        save_hotspots_overlay: bool = True
        ) -> dict:
    
    # --- 1) Load image ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Not readable image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]


    # --- 2) Edge map (Sobel magnitude) ---
    if create_edge_map:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_map = mag


    # --- 3) Local variance (sliding window) ---
    # 9x9 window (tunable)
    if create_local_variance_map:
        k = 9
        mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(k, k))
        mean_sq = cv2.boxFilter((gray.astype(np.float32)**2), ddepth=-1, ksize=(k, k))
        var_map = np.maximum(mean_sq - mean**2, 0.0)


    # --- 4) High-frequency energy (hi-pass) ---
    # Blurred - original -> high-freq approx (or use Laplacian)
    if create_high_freq_map:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2, sigmaY=2)
        hf_map = cv2.absdiff(gray, blur).astype(np.float32)


    # --- 5) Normalizes each map on [0,1] safely (percentiles) ---
    edge_n = robust_normalize(edge_map)
    var_n  = robust_normalize(var_map)
    hf_n   = robust_normalize(hf_map)


    # --- 6) Weigthed fusion into a heatmap (weights are correlated with defects) ---
    w_edge, w_var, w_hf = 0.4, 0.3, 0.3
    heat = (w_edge * edge_n + w_var * var_n + w_hf * hf_n).astype(np.float32)


    # --- 7) Show and save overlay ---
    heat_color = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(img, 0.75, heat_color, 0.35, 0)

    hotspots_heat_path = compose_filename(image_path, "03A_hotspots_heat")
    if save_hotspots_heat:
        cv2.imwrite(hotspots_heat_path, heat_color,)

    hotspots_overlay_path = compose_filename(image_path, "03B_hotspots_overlay")
    if save_hotspots_overlay:
        cv2.imwrite(hotspots_overlay_path, overlay)

    return {"hotspots_heat_path": hotspots_heat_path, "hotspots_overlay_path": hotspots_overlay_path}