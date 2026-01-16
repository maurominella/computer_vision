# ---------- Imports ----------
import os, json, requests
from .utils import compose_filename


#region helper functions
def iou(a, b):
    """simple dedup on overlapping boxes, same as SDK path"""
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a["w"]*a["h"] + b["w"]*b["h"] - inter
    return 0.0 if union == 0 else inter / union

def with_trailing_slash(url: str) -> str:
    """Ensure VISION_ENDPOINT has a trailing slash; add it if missing."""
    return url if url.endswith("/") else url + "/"

def get_payload_filename(image_path: str) -> str:
    """
    Given an image path like ./images/1. ARTWORK COLLISION/J74Q10KAUG0-G6N3.png,
    create its payload file name as artifacts/J74Q10KAUG0-ROI.json
    """
    # Extract the filename from the path
    filename = os.path.basename(image_path)
    # Remove the extension
    base_name = os.path.splitext(filename)[0]
    # Split by '-' and take the first part (e.g., J74Q10KAUG0 from J74Q10KAUG0-G6N3)
    prefix = base_name.split('-')[0]
    # Create the payload filename
    return f"artifacts/{prefix}-ROI.json"

# Parse REST JSON to the same payload schema used by the SDK path
def to_bbox_dict(bb):
    return {"x": int(bb["x"]), "y": int(bb["y"]), "w": int(bb["w"]), "h": int(bb["h"])}

#endregion

def roi_identification(
        image_path: str,
        VISION_ENDPOINT: str, 
        VISION_KEY: str,
        features: str = "Caption,Objects,Tags,DenseCaptions",
        api_version: str="2023-10-01",
        save_payload: bool = True,
        keep_confidence: float = 0.6,
        limit_proposals: int = 12
        ) -> dict:
    """
    Identify ROIs in an image using Azure Computer Vision REST API.
    """
        
    endpoint = with_trailing_slash(VISION_ENDPOINT)
    url = f"{endpoint}computervision/imageanalysis:analyze?api-version={api_version}&features={features}&model-version=latest"
    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream"
    }
    with open(image_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f.read(), timeout=60)
    
    resp.raise_for_status()
    data = resp.json()

    payload = {"context": {}, "proposals": [], "global_tags": []}

    # Caption
    cap = data.get("captionResult")
    if cap and cap.get("text"):
        payload["context"] = {"image_path": image_path, "text": cap["text"], "confidence": float(cap.get("confidence", 0.0))}

    # Dense Captions
    dense = data.get("denseCaptionsResult", {})
    for v in dense.get("values", []):
        bb = v.get("boundingBox", {})
        payload["proposals"].append({
            "source": "dense_captions",
            "text": v.get("text", ""),
            "confidence": float(v.get("confidence", 0.0)),
            "bbox": to_bbox_dict(bb)
        })

    # Objects
    objs = data.get("objectsResult", {})
    for o in objs.get("values", []):
        bb = o.get("boundingBox", {})
        name = o.get("name") or (o.get("tags", [{}])[0].get("name") if o.get("tags") else "object")
        payload["proposals"].append({
            "source": "objects",
            "text": name,
            "confidence": float(o.get("confidence", 0.0)),
            "bbox": to_bbox_dict(bb)
        })

    # Tags
    tags = data.get("tagsResult", {})
    for t in tags.get("values", []):
        payload["global_tags"].append({
            "name": t.get("name", ""),
            "confidence": float(t.get("confidence", 0.0))
        })


    proposals_sorted = sorted(payload["proposals"], key=lambda p: (p["source"] != "dense_captions", -p["confidence"]))
    selected = []
    for p in proposals_sorted:
        if not selected:
            selected.append(p); continue
        if max(iou(p["bbox"], s["bbox"]) for s in selected) < keep_confidence:
            selected.append(p)

    if limit_proposals > 0:
        payload["proposals"] = selected[:limit_proposals]

    if save_payload:        
        with open(compose_filename(image_path, "01_ROI", "json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
    
    return payload