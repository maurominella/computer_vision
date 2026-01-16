from PIL import Image, ImageDraw, ImageFont
import json


def draw_bboxes(original_path:str, result_json:str, out_path:str) -> str:
    data = result_json if isinstance(result_json, dict) else json.loads(result_json)
    im = Image.open(original_path).convert("RGB")
    w, h = im.size
    dr = ImageDraw.Draw(im)
    color = (255, 0, 0)

    for b in data.get("bboxes", []):
        x0 = int(b["x_min"] * w)
        y0 = int(b["y_min"] * h)
        x1 = int(b["x_max"] * w)
        y1 = int(b["y_max"] * h)
        for t in range(4):  # thickness
            dr.rectangle([x0-t,y0-t,x1+t,y1+t], outline=color)
        label = f'{b["label"]} {b["confidence"]:.2f}'
        # label bg
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(0.025*h)))
        except:
            font = ImageFont.load_default()
        tw, th = dr.textbbox((0,0), label, font=font)[2:]
        pad = 4
        lx, ly = x0, max(0, y0 - th - 2*pad)
        dr.rectangle([lx-pad, ly-pad, lx+tw+pad, ly+th+pad], fill=color)
        dr.text((lx, ly), label, fill=(255,255,255), font=font)

    im.save(out_path)
    return out_path