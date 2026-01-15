# Overlay regions (bbox + captions) on the original image.
# Inputs:
#   - image_path                 : local path to the image (already set in env)
#   - artifacts/proposals.json  : payload with context, proposals, global_tags
#
# Outputs:
#   - artifacts/roi_overlay.png         : overlay with rectangles only
#   - artifacts/roi_overlay_labels.png  : overlay with rectangles + labels



# ---------- Helpers ----------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def draw_bbox(draw, bbox, color=(255, 0, 0), width=3):
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=width)

def wrap_caption(text, max_chars=36):
    # Simple word-wrap for captions
    text = text or ""
    return "\n".join(textwrap.wrap(text, width=max_chars)) if text else ""

def measure_multiline(draw, text, font, padding=4):
    """
    Measure a multiline text box using draw.textbbox().
    Returns (box_w, box_h) without including the (x,y) origin,
    plus the per-line heights to help with vertical layout.
    """
    lines = text.split("\n") if text else [""]
    line_heights = []
    max_w = 0
    total_h = 0
    for ln in lines:
        # textbbox returns (left, top, right, bottom)
        l, t, r, b = draw.textbbox((0, 0), ln, font=font)
        w = r - l
        h = b - t
        max_w = max(max_w, w)
        total_h += h
        line_heights.append(h)
    box_w = max_w + 2 * padding
    box_h = total_h + 2 * padding
    return box_w, box_h, line_heights

def draw_label(draw, xy, text, bg_color, fg_color=(255, 255, 255), font=None, padding=4):
    """
    Draw a filled label box at (x, y) with word-wrapped text.
    Keeps the label within image bounds.
    """
    if font is None:
        font = ImageFont.load_default()
    # Measure text
    box_w, box_h, line_heights = measure_multiline(draw, text, font, padding=padding)

    x, y = xy
    x = clamp(x, 0, max(0, W - box_w))
    y = clamp(y, 0, max(0, H - box_h))

    # Background box
    draw.rectangle([(x, y), (x + box_w, y + box_h)], fill=bg_color)

    # Render text line-by-line
    tx = x + padding
    ty = y + padding
    for i, ln in enumerate(text.split("\n")):
        draw.text((tx, ty), ln, fill=fg_color, font=font)
        ty += line_heights[i]

# (Optional) filter out near full-frame boxes (>85% area)
def area_ratio(b):
    return (b["w"] * b["h"]) / float(W * H + 1e-9)


def roi(str image_path, dict proposals):
    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size    
        
    # Color map by source
    SRC_COLOR = {
        "dense_captions": (255, 0, 0),    # red
        "objects":        (0, 128, 255),  # blue
    }
    LABEL_BG = {
        "dense_captions": (255, 0, 0),
        "objects":        (0, 128, 255),
    }
    
    # Try to use a truetype font if available, otherwise fallback
    try:
        # Adjust to a valid TTF on your system if you prefer a specific font
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()
    
    # Prepare canvases
    overlay_boxes = img.copy()
    overlay_labels = img.copy()
    draw_boxes = ImageDraw.Draw(overlay_boxes)
    draw_labels = ImageDraw.Draw(overlay_labels)
    
    proposals = payload.get("proposals", [])
    
    proposals_for_draw = [p for p in proposals if area_ratio(p["bbox"]) <= 0.85]
    
    # Draw rectangles only
    for p in proposals_for_draw:
        src = p.get("source", "dense_captions")
        color = SRC_COLOR.get(src, (255, 0, 0))
        draw_bbox(draw_boxes, p["bbox"], color=color, width=3)
    
    # Draw rectangles + labels
    for p in proposals_for_draw:
        src = p.get("source", "dense_captions")
        color = SRC_COLOR.get(src, (255, 0, 0))
        bg = LABEL_BG.get(src, (255, 0, 0))
    
        # Rectangle
        draw_bbox(draw_labels, p["bbox"], color=color, width=3)
    
        # Build label text
        cap = p.get("text", "")
        conf = p.get("confidence", 0.0)
        label_text = f"[{src}] conf={conf:.2f}\n" + wrap_caption(cap, max_chars=38)
    
        # Place label near the top-left corner of the box
        bx, by, bw, bh = p["bbox"]["x"], p["bbox"]["y"], p["bbox"]["w"], p["bbox"]["h"]
        label_x = bx + 3
        # Try above the box; if out-of-bounds, place inside
        label_y = by - 44
        if label_y < 0:
            label_y = by + 3
    
        draw_label(draw_labels, (label_x, label_y), label_text, bg_color=bg, fg_color=(255, 255, 255), font=font, padding=4)
    
    # Save outputs
    os.makedirs("artifacts", exist_ok=True)
    overlay_boxes.save("artifacts/roi_overlay.png")
    overlay_labels.save("artifacts/roi_overlay_labels.png")
    
    print("Saved:")
    print(" - artifacts/roi_overlay.png")
    print(" - artifacts/roi_overlay_labels.png")